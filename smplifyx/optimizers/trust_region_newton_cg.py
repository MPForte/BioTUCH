# -*- coding: utf-8 -*-

# Debug version of Trust Region Newton CG with comprehensive mathematical verification
# This version adds extensive debugging to verify correctness of the implementation

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from typing import NewType, List, Tuple, Optional, Dict, Any, Union
import warnings
import time
import numpy as np

import torch
import torch.optim as optim
import torch.autograd as autograd

import math

Tensor = NewType('Tensor', torch.Tensor)


class TrustRegionNewtonCG(optim.Optimizer):
    def __init__(self, params: List[Tensor],
                 max_trust_radius: float = 1000, 
                 initial_trust_radius: float = 0.05, 
                 eta: float = 0.15, 
                 gtol: float = 1e-05, 
                 debug_level: int = 0,  # Increased default debug level
                 hessian_reg: float = 1e-6,
                 max_cg_iters: int = 50,
                 min_trust_radius: float = 1e-7,
                 verify_math: bool = True,  # NEW: Enable mathematical verification
                 **kwargs) -> None:
        
        defaults = dict()
        super(TrustRegionNewtonCG, self).__init__(params, defaults)

        self.steps = 0
        self.max_trust_radius = max_trust_radius
        self.initial_trust_radius = initial_trust_radius
        self.eta = eta
        self.gtol = gtol
        self.debug_level = debug_level
        self.hessian_reg = hessian_reg
        self.max_cg_iters = max_cg_iters
        self.min_trust_radius = min_trust_radius
        self.verify_math = verify_math
        self._params = self.param_groups[0]['params']
        
        # Enhanced statistics for debugging
        self.stats = {
            'iterations': 0,
            'accepted_steps': 0,
            'rejected_steps': 0,
            'trust_radius_resets': 0,
            'cg_iterations': [],
            'improvement_ratios': [],
            'step_norms': [],
            'gradient_norms': [],
            'loss_values': [],
            'execution_times': [],
            'quadratic_model_errors': [],  # NEW
            'cauchy_step_comparisons': [],  # NEW
            'trust_region_violations': [],  # NEW
        }

        self.stagnation_window = 5
        self.stagnation_tol = 1e-6
        self.has_converged = False
        self.convergence_reason = None
        self.max_radius_reductions = 8
        self.radius_reduction_count = 0

    def debug_print(self, level: int, *args, **kwargs):
        """Print debug information if debug_level is high enough"""
        if self.debug_level >= level:
            print(f"[TR-DEBUG-L{level}]", *args, **kwargs)

    def _verify_gradient_computation(self, closure, flat_grad, epsilon=1e-5):
        """Verify gradient computation using finite differences"""
        if not self.verify_math:
            return True
            
        self.debug_print(3, "=== GRADIENT VERIFICATION ===")
        
        # Store original parameters
        param_vector = torch.cat([p.data.view(-1) for p in self._params])
        
        # Sample a few random indices for verification (not all - too expensive)
        n_samples = min(10, len(param_vector))
        indices = torch.randperm(len(param_vector))[:n_samples]
        
        max_rel_error = 0.0
        for i in indices:
            orig_val = param_vector[i].item()
            param_epsilon = max(abs(orig_val) * 0.01, epsilon)

            # Positive perturbation
            param_vector[i] = orig_val + param_epsilon
            self._load_param_vector(param_vector)
            f_plus = closure(backward=False).item()

            # Negative perturbation
            param_vector[i] = orig_val - param_epsilon
            self._load_param_vector(param_vector)
            f_minus = closure(backward=False).item()

            # Restore original
            param_vector[i] = orig_val

            # Numerical gradient
            numeric_grad = (f_plus - f_minus) / (2 * param_epsilon)
            analytic_grad = flat_grad[i].item()
            
            # Relative error
            rel_error = abs(numeric_grad - analytic_grad) / (abs(numeric_grad) + abs(analytic_grad) + 1e-8)
            max_rel_error = max(max_rel_error, rel_error)
            
            self.debug_print(4, f"  Param {i}: numeric={numeric_grad:.6e}, analytic={analytic_grad:.6e}, rel_error={rel_error:.3e}")

        # Restore original parameters
        self._load_param_vector(param_vector)
        for p in self._params:
            p.requires_grad_(True)
            
        self.debug_print(3, f"Gradient verification: max_rel_error = {max_rel_error:.3e}")
        return max_rel_error < 1e-3

    def _verify_hessian_vector_product(self, gradient, direction, hvp_result):
        """Verify Hessian-vector product using finite differences"""
        if not self.verify_math:
            return True
            
        self.debug_print(3, "=== HESSIAN-VECTOR PRODUCT VERIFICATION ===")
        
        epsilon = 1e-5
        
        # Store original parameters
        original_params = [p.data.clone() for p in self._params]
        
        # Compute gradient at x + ε*d
        start_idx = 0
        for p in self._params:
            numel = p.numel()
            direction_slice = direction[start_idx:start_idx + numel].view_as(p)
            p.data.add_(direction_slice, alpha=epsilon)
            start_idx += numel
        
        # Ensure gradients are computed
        for p in self._params:
            p.requires_grad_(True)
        
        # Get gradient at perturbed point
        dummy_loss = sum(torch.sum(p) for p in self._params)  # Dummy loss for gradient computation
        dummy_loss.backward()
        grad_plus = torch.cat([p.grad.view(-1) for p in self._params])
        
        # Clear gradients and restore parameters
        for p, orig_data in zip(self._params, original_params):
            p.grad = None
            p.data.copy_(orig_data)
        
        # Compute gradient at x - ε*d  
        start_idx = 0
        for p in self._params:
            numel = p.numel()
            direction_slice = direction[start_idx:start_idx + numel].view_as(p)
            p.data.add_(direction_slice, alpha=-epsilon)
            start_idx += numel
            
        # Get gradient at negatively perturbed point
        dummy_loss = sum(torch.sum(p) for p in self._params)
        dummy_loss.backward()
        grad_minus = torch.cat([p.grad.view(-1) for p in self._params])
        
        # Restore original parameters
        for p, orig_data in zip(self._params, original_params):
            p.grad = None
            p.data.copy_(orig_data)
            p.requires_grad_(True)
        
        # Finite difference approximation of Hessian-vector product
        hvp_numerical = (grad_plus - grad_minus) / (2 * epsilon)
        
        # Compare with analytical result
        rel_error = torch.norm(hvp_numerical - hvp_result) / (torch.norm(hvp_numerical) + torch.norm(hvp_result) + 1e-8)
        
        self.debug_print(3, f"HVP verification: relative error = {rel_error.item():.3e}")
        self.debug_print(4, f"  Numerical norm: {torch.norm(hvp_numerical).item():.6e}")
        self.debug_print(4, f"  Analytical norm: {torch.norm(hvp_result).item():.6e}")
        
        return rel_error.item() < 1e-2  # More lenient for second-order methods

    def _compute_cauchy_step(self, gradient, trust_radius):
        """Compute the Cauchy step for comparison with CG solution"""
        grad_norm = torch.norm(gradient)
        
        if grad_norm == 0:
            return torch.zeros_like(gradient)
        
        # Compute Hessian-vector product for the gradient direction
        hvp = self._compute_hessian_vector_product(gradient, -gradient)
        curvature = torch.dot(gradient, hvp)
        
        if curvature <= 0:
            # Negative curvature - go to trust region boundary
            return -(trust_radius / grad_norm) * gradient
        else:
            # Positive curvature - use Newton step or trust region boundary
            newton_step_length = (grad_norm ** 2) / curvature
            if newton_step_length >= trust_radius:
                return -(trust_radius / grad_norm) * gradient
            else:
                return -(newton_step_length / grad_norm) * gradient

    def _verify_trust_region_constraint(self, step, trust_radius):
        """Verify that the step satisfies the trust region constraint"""
        step_norm = torch.norm(step).item()
        constraint_violation = max(0, step_norm - trust_radius)
        
        self.stats['trust_region_violations'].append(constraint_violation)
        
        if constraint_violation > 1e-6:
            self.debug_print(2, f"WARNING: Trust region constraint violated by {constraint_violation.item():.6e}")
            self.debug_print(2, f"  Step norm: {step_norm:.6e}, Trust radius: {trust_radius.item():.6e}")
            return False
        
        self.debug_print(3, f"Trust region constraint satisfied: ||step|| = {step_norm:.6e} <= {trust_radius.item():.6e}")
        return True

    def _verify_quadratic_model(self, step, loss, gradient, hvp, actual_loss_change):
        """Verify the quadratic model prediction accuracy"""
        if not self.verify_math:
            return
            
        linear_term = torch.dot(gradient, step).item()
        quadratic_term = 0.5 * torch.dot(step, hvp).item()
        predicted_change = linear_term + quadratic_term
        
        model_error = abs(predicted_change - actual_loss_change) / (abs(actual_loss_change) + 1e-8)
        self.stats['quadratic_model_errors'].append(model_error.item() if hasattr(model_error, 'item') else model_error)

        
        self.debug_print(3, "=== QUADRATIC MODEL VERIFICATION ===")
        self.debug_print(3, f"  Actual loss change: {actual_loss_change.item():.6e}")
        self.debug_print(3, f"  Linear term: {linear_term:.6e}")
        self.debug_print(3, f"  Quadratic term: {quadratic_term:.6e}")
        self.debug_print(3, f"  Predicted change: {predicted_change:.6e}")
        self.debug_print(3, f"  Model error: {model_error:.3e}")
        
        if model_error > 0.5:
            self.debug_print(2, f"WARNING: Large quadratic model error: {model_error:.3e}")

    def _load_param_vector(self, vector):
        """Load parameter vector into the model parameters"""
        start_idx = 0
        for p in self._params:
            original_requires_grad = p.requires_grad
            
            with torch.no_grad():
                p.data.copy_(p.detach().clone())
            
            p.requires_grad_(original_requires_grad)

            numel = p.numel()
            with torch.no_grad():
                p.data.copy_(vector[start_idx:start_idx + numel].view_as(p))

            p.requires_grad_(original_requires_grad)
            start_idx += numel

    @torch.enable_grad()
    def _compute_hessian_vector_product(self, gradient: Tensor, p: Tensor, regularize: bool = True) -> Tensor:
        """Compute Hessian-vector product H·p where H = ∇²f and p is a vector."""
        p_reg = p.clone()
        
        # Compute directional derivative: (∇f)ᵀ·p
        directional_derivative = torch.sum(gradient * p_reg)
        
        try:
            # Differentiate again to get H·p
            hess_vp = autograd.grad(
                directional_derivative,
                self._params,
                retain_graph=True
            )
            
            flat_hvp = torch.cat([torch.flatten(vp) for vp in hess_vp], dim=-1)
            
            # Add regularization for numerical stability
            if regularize:
                regularized_hvp = flat_hvp + self.hessian_reg * p_reg
                
                # Verify the HVP computation
                if self.verify_math and self.steps % 10 == 0:  # Only verify occasionally
                    self._verify_hessian_vector_product(gradient, p_reg, regularized_hvp)
                
                return regularized_hvp
            else:
                return flat_hvp
                
        except RuntimeError as e:
            self.debug_print(2, f"Hessian-vector product error: {e}, using regularized fallback")
            return self.hessian_reg * p_reg

    def _gather_flat_grad(self) -> Tensor:
        """Concatenates all gradients into a single gradient vector"""
        views = []
        for p in self._params:
            if p.grad is None:
                view = p.data.new(p.data.numel()).zero_()
            elif p.grad.data.is_sparse:
                view = p.grad.to_dense().view(-1)
            else:
                view = p.grad.view(-1)
            views.append(view)
        output = torch.cat(views, 0)
        return output

    @torch.no_grad() 
    def _validate_descent_direction(self, step: Tensor, gradient: Tensor) -> bool:
        """Validate that the step is a descent direction using dot product sign"""
        directional_derivative = torch.sum(gradient * step)
        is_descent = directional_derivative.item() < 0
        
        self.debug_print(3, f"Descent direction check: g^T*p = {directional_derivative.item():.6e}")
        
        if not is_descent:
            self.debug_print(2, f"WARNING: Non-descent direction detected!")
        
        return is_descent

    @torch.no_grad()
    def _improvement_ratio(self, p, start_loss, gradient, closure):
        """Calculates the ratio of the actual to the expected improvement"""
        
        self.debug_print(3, "=== IMPROVEMENT RATIO CALCULATION ===")
        
        # Store original state for ALL parameters
        original_state = []
        for param in self._params:
            original_state.append({
                'requires_grad': param.requires_grad,
                'data': param.data.clone()
            })
        
        # Calculate model predictions before parameter updates
        model_at_current = start_loss
        hess_vp = self._compute_hessian_vector_product(gradient, p)
        model_at_step = self._quad_model(p, start_loss, gradient, hess_vp)
        predicted_reduction = model_at_current - model_at_step
        
        self.debug_print(3, f"  Current loss: {model_at_current.item():.6e}")
        self.debug_print(3, f"  Predicted loss at step: {model_at_step.item():.6e}")
        self.debug_print(3, f"  Predicted reduction: {predicted_reduction.item():.6e}")
        
        # Apply parameter updates using .data to preserve leaf status
        start_idx = 0
        for param in self._params:
            num_els = param.numel()
            curr_upd = p[start_idx:start_idx + num_els]
            param.data.add_(curr_upd.view_as(param))
            start_idx += num_els

        # Restore requires_grad status after parameter updates
        for param, state in zip(self._params, original_state):
            param.requires_grad_(state['requires_grad'])

        # Evaluate function at new point
        new_loss = closure(backward=False)
        
        self.debug_print(3, f"  Actual loss at step: {new_loss.item():.6e}")
        
        # Restore requires_grad status again after closure call
        for param, state in zip(self._params, original_state):
            param.requires_grad_(state['requires_grad'])
        
        # Calculate improvement ratio
        actual_reduction = start_loss - new_loss
        
        self.debug_print(3, f"  Actual reduction: {actual_reduction.item():.6e}")
        
        # Verify quadratic model accuracy
        if self.verify_math:
            self._verify_quadratic_model(p, start_loss, gradient, hess_vp, actual_reduction)
        
        # Handle edge cases and convert to float for calculations
        actual_reduction_val = actual_reduction.item()
        predicted_reduction_val = predicted_reduction.item()
        
        epsilon = 1e-10
        if abs(predicted_reduction_val) < epsilon:
            ratio = 1.0 if actual_reduction_val > 0 else 0.0
            self.debug_print(3, f"  Near-zero predicted reduction, using ratio = {ratio}")
        else:
            if predicted_reduction_val < 0:
                self.debug_print(2, f"WARNING: Negative predicted reduction {predicted_reduction_val:.6e}")
                ratio = 0.5 if actual_reduction_val > 0 else 0.0
            else:
                ratio = actual_reduction_val / predicted_reduction_val
        
        # FIX: ratio is already a Python float, don't call .item() on it
        self.debug_print(3, f"  Final improvement ratio: {ratio:.6e}")
        
        # Log for statistics (ratio is already a float)
        self.stats['improvement_ratios'].append(ratio)
        
        # Return as tensor for consistency with the rest of the optimizer
        return torch.tensor(ratio, device=gradient.device, dtype=gradient.dtype)

    @torch.no_grad()
    def _quad_model(self, p: Tensor, loss: float, gradient: Tensor, hess_vp: Tensor) -> float:
        """Returns the value of the local quadratic approximation"""
        linear_term = torch.sum(gradient * p)
        quadratic_term = 0.5 * torch.sum(p * hess_vp)
        model_value = loss + linear_term + quadratic_term
        
        self.debug_print(4, f"    Quadratic model: loss={loss.item():.6e}, linear={linear_term:.6e}, quad={quadratic_term:.6e}")
        
        return model_value

    @torch.no_grad()
    def calc_boundaries(self, iterate: Tensor, direction: Tensor, trust_radius: float) -> Tuple[Tensor, Tensor]:
        """Calculates the offset to the boundaries of the trust region"""
        device = iterate.device

        # Calculate coefficients for quadratic equation: a*t^2 + b*t + c = 0
        a = torch.sum(direction ** 2)
        b = 2 * torch.sum(direction * iterate)
        c = torch.sum(iterate ** 2) - trust_radius ** 2
        
        discriminant = b * b - 4 * a * c
        
        self.debug_print(4, f"    Boundary calc: a={a.item():.6e}, b={b.item():.6e}, c={c.item():.6e}, disc={discriminant.item():.6e}")
        
        # Ensure discriminant is positive
        if discriminant < 0:
            if abs(discriminant) < 1e-10:
                discriminant = torch.tensor(0.0, device=discriminant.device)
                self.debug_print(3, "    Near-zero discriminant, setting to 0")
            else:
                self.debug_print(2, f"WARNING: Negative discriminant in boundary calculation: {discriminant.item():.6e}")
                # Fallback solution
                norm_direction = torch.norm(direction)
                if norm_direction > 1e-10:
                    iterate_norm = torch.norm(iterate)
                    if iterate_norm < trust_radius:
                        t = (trust_radius - iterate_norm) / norm_direction
                        return torch.tensor([t], device=iterate.device), torch.tensor([t], device=iterate.device)
                    else:
                        return torch.tensor([0.0], device=iterate.device), torch.tensor([0.001], device=iterate.device)
                else:
                    return torch.tensor([0.0], device=iterate.device), torch.tensor([0.0], device=iterate.device)

        sqrt_discriminant = torch.sqrt(discriminant)
        
        # Avoid division by zero
        denom = 2 * a
        if abs(denom.item()) < 1e-10:
            self.debug_print(3, "    Near-zero denominator in boundary calculation")
            return torch.tensor([0.0], device=device), torch.tensor([0.0], device=device)
            
        ta = (-b + sqrt_discriminant) / denom
        tb = (-b - sqrt_discriminant) / denom
        
        self.debug_print(4, f"    Boundary solutions: ta={ta.item():.6e}, tb={tb.item():.6e}")
        
        if ta.item() < tb.item():
            return torch.tensor([ta.item()], device=device), torch.tensor([tb.item()], device=device)
        else:
            return torch.tensor([tb.item()],  device=device), torch.tensor([ta.item()], device=device)

    @torch.no_grad()
    def _solve_trust_reg_subproblem(self, loss: float, flat_grad: Tensor, trust_radius: float) -> Tuple[Tensor, bool]:
        """Solves the quadratic subproblem in the trust region"""
        
        self.debug_print(3, "=== TRUST REGION SUBPROBLEM SOLUTION ===")
        self.debug_print(3, f"  Trust radius: {trust_radius.item():.6e}")
        self.debug_print(3, f"  Gradient norm: {torch.norm(flat_grad).item():.6e}")
        
        # Initialize
        iterate = torch.zeros_like(flat_grad, requires_grad=False)
        residual = flat_grad.detach().clone()
        direction = -residual
        
        # Tolerance for CG stopping condition
        jac_mag = torch.norm(flat_grad).item()
        tolerance = min(0.5, math.sqrt(jac_mag)) * jac_mag

        self.debug_print(3, f"  CG tolerance: {tolerance:.6e}")

        # If gradient is small enough, exit early
        if jac_mag <= tolerance:
            self.debug_print(3, "  Gradient small enough, returning zero step")
            return iterate, False

        # Compute Cauchy step for comparison
        if self.verify_math:
            cauchy_step = self._compute_cauchy_step(flat_grad, trust_radius)
            cauchy_norm = torch.norm(cauchy_step).item()
            self.debug_print(3, f"  Cauchy step norm: {cauchy_norm:.6e}")

        cg_iterations = 0
        
        # Main CG loop
        while cg_iterations < self.max_cg_iters:
            cg_iterations += 1
            
            self.debug_print(4, f"    CG iteration {cg_iterations}")
            self.debug_print(4, f"      Iterate norm: {torch.norm(iterate).item():.6e}")
            self.debug_print(4, f"      Residual norm: {torch.norm(residual).item():.6e}")
            
            # Calculate the Hessian-Vector product
            hessian_vec_prod = self._compute_hessian_vector_product(flat_grad, direction)

            hevp_dot_prod = torch.sum(direction * hessian_vec_prod)
            
            self.debug_print(4, f"      Direction^T * H * direction: {hevp_dot_prod.item():.6e}")

            # Check for non-positive curvature
            if hevp_dot_prod.item() <= 0:
                self.debug_print(3, f"  Non-positive curvature at CG iteration {cg_iterations}")
                
                # Find boundary intersections
                ta, tb = self.calc_boundaries(iterate, direction, trust_radius)
                
                # Compute points at both intersections
                pa = iterate + ta * direction
                pb = iterate + tb * direction

                # Calculate the point on the boundary with the smallest model value
                hess_vp_a = self._compute_hessian_vector_product(flat_grad, pa)
                hess_vp_b = self._compute_hessian_vector_product(flat_grad, pb)
                
                bound1_val = self._quad_model(pa, loss, flat_grad, hess_vp_a)
                bound2_val = self._quad_model(pb, loss, flat_grad, hess_vp_b)
                
                self.debug_print(3, f"  Boundary option A: model_val={bound1_val.item():.6e}, norm={torch.norm(pa).item():.6e}")
                self.debug_print(3, f"  Boundary option B: model_val={bound2_val.item():.6e}, norm={torch.norm(pb).item():.6e}")
                
                self.stats['cg_iterations'].append(cg_iterations)
                
                result = pa if bound1_val.item() < bound2_val.item() else pb
                self._verify_trust_region_constraint(result, trust_radius)
                return result, True

            # The squared euclidean norm of the residual needed for the CG update
            residual_sq_norm = torch.sum(residual * residual)

            # Compute the step size for the CG algorithm
            epsilon = 1e-10
            if abs(hevp_dot_prod.item()) < epsilon:
                self.debug_print(3, f"  Near-zero curvature, stopping CG")
                self.stats['cg_iterations'].append(cg_iterations)
                return iterate, False
                
            cg_step_size = residual_sq_norm / hevp_dot_prod
            self.debug_print(4, f"      CG step size: {cg_step_size.item():.6e}")

            # Update the point
            next_iterate = iterate + cg_step_size * direction

            # Check if the step takes us outside the trust region
            iterate_norm = torch.norm(next_iterate)

            self.debug_print(4, f"      Next iterate norm: {iterate_norm.item():.6e}")

            # If outside trust region, project to boundary
            if iterate_norm.item() >= trust_radius:
                self.debug_print(3, f"  Step would exceed trust region, projecting to boundary")
                ta, tb = self.calc_boundaries(iterate, direction, trust_radius)
                t_boundary = tb if tb.item() > 0 else ta
                p_boundary = iterate + t_boundary * direction
                
                self.debug_print(3, f"  Boundary step: t={t_boundary.item():.6e}, norm={torch.norm(p_boundary).item():.6e}")
                
                self.stats['cg_iterations'].append(cg_iterations)
                self._verify_trust_region_constraint(p_boundary, trust_radius)
                return p_boundary, True

            # Update the residual
            next_residual = residual + cg_step_size * hessian_vec_prod
            
            # Check for NaN values
            if torch.isnan(next_residual).any():
                self.debug_print(2, f"WARNING: NaN detected in residual at CG iteration {cg_iterations}")
                self.stats['cg_iterations'].append(cg_iterations)
                return iterate, False
                
            # Check convergence
            residual_norm = torch.norm(next_residual).item()
            self.debug_print(4, f"      New residual norm: {residual_norm:.6e}")
            
            if residual_norm < tolerance:
                self.debug_print(3, f"  CG converged at iteration {cg_iterations}")
                self.stats['cg_iterations'].append(cg_iterations)
                self._verify_trust_region_constraint(next_iterate, trust_radius)
                return next_iterate, False

            # Compute beta for the new search direction
            beta = torch.sum(next_residual ** 2) / residual_sq_norm
            next_direction = -next_residual + beta * direction
            
            self.debug_print(4, f"      Beta: {beta.item():.6e}")
            
            if torch.isnan(next_direction).any():
                self.debug_print(2, f"WARNING: NaN detected in direction at CG iteration {cg_iterations}")
                self.stats['cg_iterations'].append(cg_iterations)
                return iterate, False

            # Update for next iteration
            iterate = next_iterate
            residual = next_residual
            direction = next_direction
            
        # Maximum iterations reached
        self.debug_print(3, f"  CG reached maximum iterations ({self.max_cg_iters})")
        self.stats['cg_iterations'].append(cg_iterations)
        self._verify_trust_region_constraint(iterate, trust_radius)
        return iterate, False
        
    def step(self, closure) -> float:
        """Performs a single optimization step."""
        
        # Check if already converged
        if self.has_converged:
            return closure(backward=False)
        
        iteration_start_time = time.time()
        self.stats['iterations'] += 1
        
        self.debug_print(2, f"\n{'='*60}")
        self.debug_print(2, f"TRUST REGION ITERATION {self.stats['iterations']}")
        self.debug_print(2, f"{'='*60}")
        
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=UserWarning)
            starting_loss = closure(backward=True)

        self.debug_print(2, f"Starting loss: {starting_loss.item():.6e}")

        # Check for loss stagnation
        if len(self.stats['loss_values']) >= self.stagnation_window:
            recent_losses = self.stats['loss_values'][-self.stagnation_window:]
            loss_range = max(recent_losses) - min(recent_losses)
            self.debug_print(3, f"Loss stagnation check: range over {self.stagnation_window} iterations = {loss_range:.6e}")
            
            if loss_range < self.stagnation_tol:
                self.has_converged = True
                self.convergence_reason = f"Loss stagnated for {self.stagnation_window} iterations (change < {self.stagnation_tol})"
                self.debug_print(1, f"CONVERGED: {self.convergence_reason}")
                return starting_loss
                
        # Save loss for tracking
        self.stats['loss_values'].append(starting_loss.item())

        flat_grad = self._gather_flat_grad()
        grad_norm = torch.norm(flat_grad).item()
        self.stats['gradient_norms'].append(grad_norm)
        
        self.debug_print(2, f"Gradient norm: {grad_norm:.6e}")
        
        # Verify gradient computation occasionally
        if self.verify_math and self.steps % 20 == 0:
            self._verify_gradient_computation(closure, flat_grad)

        # Initialize state if first step
        state = self.state
        if len(state) == 0:
            state['trust_radius'] = torch.full([1],
                                            self.initial_trust_radius,
                                            dtype=flat_grad.dtype,
                                            device=flat_grad.device)
            state['reset_count'] = 0
            state['iteration'] = 0
            self.debug_print(2, f"Initializing trust radius: {self.initial_trust_radius:.6e}")
            
        # Increment iteration counter
        state['iteration'] = state.get('iteration', 0) + 1
        
        trust_radius = state['trust_radius']
        self.debug_print(2, f"Current trust radius: {trust_radius.item():.6e}")

        # Check if gradient is small enough to converge
        if grad_norm <= self.gtol:
            self.has_converged = True
            self.convergence_reason = f"Gradient norm {grad_norm:.6e} below tolerance {self.gtol.item():.6e}"
            self.debug_print(1, f"CONVERGED: {self.convergence_reason}")
            return starting_loss

        # Solve the trust region subproblem to get the step
        param_step, hit_boundary = self._solve_trust_reg_subproblem(
            starting_loss, flat_grad, trust_radius)
        self.param_step = param_step
        
        # Track step norm
        step_norm = torch.norm(param_step).item()
        self.stats['step_norms'].append(step_norm)
        
        self.debug_print(2, f"Step norm: {step_norm:.6e}")
        self.debug_print(2, f"Hit boundary: {hit_boundary}")

        # Validate descent direction
        if not self._validate_descent_direction(param_step, flat_grad):
            self.debug_print(1, "WARNING: Non-descent direction, skipping step")
            self.stats['rejected_steps'] += 1
            return starting_loss

        # Compare with Cauchy step if verification is enabled
        if self.verify_math:
            cauchy_step = self._compute_cauchy_step(flat_grad, trust_radius.item())
            cauchy_norm = torch.norm(cauchy_step).item()
            
            # The CG step should be at least as good as the Cauchy step
            cg_model_val = self._quad_model(param_step, starting_loss, flat_grad, 
                                          self._compute_hessian_vector_product(flat_grad, param_step))
            cauchy_model_val = self._quad_model(cauchy_step, starting_loss, flat_grad,
                                              self._compute_hessian_vector_product(flat_grad, cauchy_step))
            
            self.debug_print(3, f"Model value comparison:")
            self.debug_print(3, f"  CG step model: {cg_model_val.item():.6e}")
            self.debug_print(3, f"  Cauchy step model: {cauchy_model_val.item():.6e}")
            
            if cg_model_val > cauchy_model_val + 1e-6:
                self.debug_print(2, f"WARNING: CG step is worse than Cauchy step!")
            
            self.stats['cauchy_step_comparisons'].append({
                'cg_model': cg_model_val.item(),
                'cauchy_model': cauchy_model_val.item(),
                'cg_norm': step_norm,
                'cauchy_norm': cauchy_norm
            })

        # Calculate improvement ratio
        improvement_ratio = self._improvement_ratio(
            param_step, starting_loss, flat_grad, closure)

        self.debug_print(2, f"Improvement ratio: {improvement_ratio.item():.6f}")

        # Update trust radius based on improvement ratio
        old_radius = trust_radius.item()
        
        if improvement_ratio.item() < 0.25:
            self.radius_reduction_count += 1
            
            if self.radius_reduction_count >= self.max_radius_reductions:
                # Reset to larger radius
                trust_radius.fill_(self.initial_trust_radius * 0.1)
                self.radius_reduction_count = 0
                self.debug_print(2, f"RESET: trust radius to {trust_radius.item():.6e} after {self.max_radius_reductions} reductions")
            else:
                trust_radius.mul_(0.5)
                self.debug_print(2, f"REDUCE: trust radius {old_radius:.6e} -> {trust_radius.item():.6e} (reduction #{self.radius_reduction_count})")
        else:
            # Reset counter on successful steps
            if improvement_ratio.item() > 0.75 and hit_boundary:
                self.radius_reduction_count = max(0, self.radius_reduction_count - 1)
                trust_radius.mul_(1.5).clamp_(0.0, self.max_trust_radius)
                self.debug_print(2, f"INCREASE: trust radius {old_radius:.6e} -> {trust_radius.item():.6e}")
            else:
                self.debug_print(2, f"MAINTAIN: trust radius at {trust_radius.item():.6e}")

        # Progressive reset strategy when trust radius gets too small
        if trust_radius.item() < self.min_trust_radius:
            state['reset_count'] += 1
            self.stats['trust_radius_resets'] += 1
            reset_multiplier = 2 ** state['reset_count']
            reset_value = min(self.initial_trust_radius * reset_multiplier, self.max_trust_radius)
            
            self.debug_print(1, f"EMERGENCY RESET #{state['reset_count']}: trust radius {trust_radius.item():.6e} -> {reset_value.item():.6e}")
            trust_radius.fill_(reset_value)
            
            if state['reset_count'] >= 5:
                self.debug_print(1, f"WARNING: Maximum reset attempts reached")

        # Decide whether to accept or reject the step
        step_accepted = improvement_ratio.item() > self.eta
        
        if step_accepted:
            self.debug_print(2, f"ACCEPT: Step accepted (ratio {improvement_ratio.item():.6f} > {self.eta:.6f})")
            self.stats['accepted_steps'] += 1
        else:
            self.debug_print(2, f"REJECT: Step rejected (ratio {improvement_ratio.item():.6f} <= {self.eta:.6f})")
            self.stats['rejected_steps'] += 1
            
            # Revert the step
            start_idx = 0
            for param in self._params:
                num_els = param.numel()
                curr_upd = param_step[start_idx:start_idx + num_els]
                param.data.add_(-curr_upd.view_as(param))
                start_idx += num_els

        # Print iteration summary
        self.debug_print(1, f"TR Summary: iter={state['iteration']}, radius={trust_radius.item():.2e}, " +
                        f"grad_norm={grad_norm:.1e}, ratio={improvement_ratio.item():.3f}, " +
                        f"{'ACCEPT' if step_accepted else 'REJECT'}")

        # Print statistics periodically
        if self.debug_level >= 2 and self.stats['iterations'] % 10 == 0:
            self._print_statistics()

        # Record execution time
        iteration_time = time.time() - iteration_start_time
        self.stats['execution_times'].append(iteration_time)

        self.steps += 1
        return starting_loss

    def _print_statistics(self):
        """Print comprehensive statistics about the optimization"""
        stats = self.stats
        
        self.debug_print(2, f"\n--- TRUST REGION STATISTICS (after {stats['iterations']} iterations) ---")
        self.debug_print(2, f"Accepted steps: {stats['accepted_steps']}")
        self.debug_print(2, f"Rejected steps: {stats['rejected_steps']}")
        self.debug_print(2, f"Accept rate: {stats['accepted_steps']/(stats['accepted_steps'] + stats['rejected_steps']):.1%}")
        self.debug_print(2, f"Trust radius resets: {stats['trust_radius_resets']}")
        
        if stats['cg_iterations']:
            avg_cg = np.mean(stats['cg_iterations'])
            self.debug_print(2, f"Average CG iterations: {avg_cg:.1f}")
        
        if stats['improvement_ratios']:
            avg_ratio = np.mean(stats['improvement_ratios'])
            self.debug_print(2, f"Average improvement ratio: {avg_ratio:.3f}")
        
        if stats['step_norms']:
            avg_step_norm = np.mean(stats['step_norms'])
            self.debug_print(2, f"Average step norm: {avg_step_norm:.3e}")
        
        if stats['gradient_norms']:
            current_grad_norm = stats['gradient_norms'][-1]
            self.debug_print(2, f"Current gradient norm: {current_grad_norm:.3e}")
        
        if self.verify_math:
            if stats['quadratic_model_errors']:
                avg_model_error = np.mean(stats['quadratic_model_errors'])
                self.debug_print(2, f"Average quadratic model error: {avg_model_error:.3e}")
            
            if stats['trust_region_violations']:
                violations = [v for v in stats['trust_region_violations'] if v > 1e-6]
                if violations:
                    self.debug_print(2, f"Trust region violations: {len(violations)}")
        
        self.debug_print(2, f"--- END STATISTICS ---\n")

    def get_debug_info(self):
        """Return comprehensive debug information"""
        return {
            'stats': self.stats,
            'convergence_status': {
                'has_converged': self.has_converged,
                'convergence_reason': self.convergence_reason,
                'total_iterations': self.stats['iterations']
            },
            'current_state': {
                'trust_radius': self.state.get('trust_radius', {}).item() if 'trust_radius' in self.state else None,
                'reset_count': self.state.get('reset_count', 0),
                'radius_reduction_count': self.radius_reduction_count
            }
        }