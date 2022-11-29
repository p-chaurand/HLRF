from typing import Union, Tuple, Any, Optional
import numpy as np
import eagerpy as ep
import copy
import torch

from ..devutils import flatten
from ..devutils import atleast_kd


from ..models import Model

from ..distances import l2

from ..criteria import Misclassification
from ..criteria import TargetedMisclassification

from .base import MinimizationAttack
from .base import T
from .base import get_criterion
from .base import raise_if_kwargs
from .base import verify_input_bounds


class iHL_RFAttack(MinimizationAttack):
    """Implementation of the improved Hasofer-Lind, Rackwitz-Fiessler Attack.

    Args:
        steps : Number of optimization steps.
        confidence : Confidence required for an example to be marked as adversarial (just considered in the loss to go behind the frontier).
            Controls the gap between example and decision boundary.
        tau : multiplicative decrease of linear stepsize in Armijo's rule.
        smooth : growing factor in penalization term.
        omega : Armijo's factor, factor by which we want to assure we reduce enough.
        abort_early : Stop inner search as soons as an adversarial example has been found.
        min_steps : if abort early dont stop before a minimum step.


    """

    distance = l2

    def __init__(
        self,
        steps: int = 50,
        confidence: float = 0.1,
        tau: float = 0.1,
        smooth: float = 4, #1.2,
        c_sigma_gain:float = 1.2,
        c_sigma_start:float = 0.5,
        omega: float = 10e-4,
        abort_early: bool = True,
        min_steps: int = 25,
        max_queries=None,
        d1_d2_ratio=2,
        delta=10e-5,
        eta=2,
        nhl_rf=False
    ):
        self.steps = steps
        self.confidence = confidence
        self.tau = tau
        self.omega = omega
        self.smooth = smooth
        self.abort_early = abort_early
        self.min_steps = min_steps
        self.max_queries = max_queries
        self.delta = delta
        self.eta = eta
        self.nhl_rf = nhl_rf
        
        self.queries = None
        self.gamma_speed_step = 1
        self.c_sigma_gain = c_sigma_gain
        self.targeted = False
        self.c_sigma_start = c_sigma_start
        self.d1_d2_ratio = d1_d2_ratio

    def loss_fun(self, x: ep.Tensor) -> Tuple[ep.Tensor, ep.Tensor]:

            """
            The loss used to study adverariality. If <0, then the image is adversarial
            """
            rows = range(len(self.classes))
            self.queries += 1
            mod = self.model(x)
            logits = ep.softmax(mod)

            if self.targeted:
                c_minimize = best_other_classes(logits, self.classes)
                c_maximize = self.classes  # target_classes
            else:
                c_minimize = self.classes  # labels
                c_maximize = best_other_classes(logits, self.classes)

            loss = (
                logits[rows, c_minimize] - logits[rows, c_maximize]
            ) + self.confidence
            return loss.sum(), loss

    def is_adversarial(self, x: ep.Tensor) -> ep.Tensor:
        if not (self.targeted):
            return self.model(x).argmax(axis=1) != self.classes
        else:
            return self.model(x).argmax(axis=1) == self.classes

    def run(
        self,
        model: Model,
        inputs: T,
        criterion: Union[Misclassification, TargetedMisclassification, T],
        *,
        early_stop: Optional[float] = None,
        **kwargs: Any,
    ) -> T:
        raise_if_kwargs(kwargs)
        self.queries = np.zeros(len(inputs))
        x, restore_type = ep.astensor_(inputs)
        self.x = x
        criterion_ = get_criterion(criterion)
        del inputs, criterion, kwargs
        verify_input_bounds(x, model)

        if isinstance(criterion_, Misclassification):
            self.targeted = False
            classes = criterion_.labels

        elif isinstance(criterion_, TargetedMisclassification):
            self.targeted = True
            classes = criterion_.target_classes
        else:
            raise ValueError("unsupported criterion")

        assert len(classes) == len(x)


        self.model = model
        self.classes = classes

        loss_aux_and_grad = ep.value_and_grad_fn(x, self.loss_fun, has_aux=True)
        best_advs = copy.deepcopy(x)
        best_advs_norms = ep.ones_like(classes) * ep.inf

        advs = copy.deepcopy(x)
        _, loss, gradient = loss_aux_and_grad(x)
        self.g_0 = loss
        # advs = x - 0.5 * gradient / atleast_kd(gradient.flatten(1).norms.lp(p=2, axis=1), gradient.ndim)

        found_advs = ep.zeros_like(classes).bool()
        self.c_sigma = ep.ones_like(classes) * self.c_sigma_start

        self.history_norms = [[] for k in range(len(x))]
        self.armijo_step = [[] for k in range(len(x))]
        self.is_advs_history = [[] for k in range(len(x))]
        self.perturbation_added_dist = [[] for k in range(len(x))]
        self.loss_gt_history = [[] for k in range(len(x))]
        self.loss_p_history = [[] for k in range(len(x))]
        self.query_history = [[] for k in range(len(x))]
        self.sigm_history = [[] for k in range(len(x))]
        self.armijo_respect = [[] for k in range(len(x))]
        self.norm_grad_history = [[] for k in range(len(x))]
        self.perturbation_grad_ratio_history = [[] for k in range(len(x))]
        self.true_gradient_norm = [[] for k in range(len(x))]
        self.gradient_norm = [[] for k in range(len(x))]
        self.descent_dir_norm = [[] for k in range(len(x))]
        self.d1_norm = [[] for k in range(len(x))]
        self.d2_norm = [[] for k in range(len(x))]
        self.d2_kept_norm = [[] for k in range(len(x))]

        norms = ep.astensor((advs - x).raw.flatten(1).norm(dim=1))
        true_norm_grad = gradient.flatten(1).norms.lp(p=2, axis=1)
        norm_grad = self.get_norm_grad(gradient)
        logits_gt, logits_p = self.get_logits(advs)
        for i in range(len(x)):
            self.history_norms[i].append(float(norms[i].raw.cpu()))
            self.is_advs_history[i].append(0)
            self.loss_gt_history[i].append(logits_gt[i].raw.cpu())
            self.loss_p_history[i].append(logits_p[i].raw.cpu())
            self.gradient_norm[i].append(norm_grad[i].raw.cpu())
            self.true_gradient_norm[i].append(true_norm_grad[i].raw.cpu())


            self.armijo_step[i].append(-1)
            self.perturbation_added_dist[i].append(-1)
            self.query_history[i].append(0)
            self.descent_dir_norm[i].append(-1)

            self.d1_norm[i].append(-1)
            self.d2_norm[i].append(-1)
            self.d2_kept_norm[i].append(-1)
        self.loss_under = loss < 0.2

        for step in range(self.steps):

            it_query_start =self.queries
            self.queries += 1
            _, loss, gradient = loss_aux_and_grad(advs)
            self.loss_under = ep.logical_or(self.loss_under,  loss < 0.2)

            perturbation = advs - x

            desc_dir = self.descent_dir(perturbation, gradient, loss)
            desc_dir_norm = desc_dir.flatten(1).norms.lp(p=2, axis=1)
            desc_dir /= atleast_kd(desc_dir_norm, desc_dir.ndim)

            stepsize = self.pred_stepsize(x=x, perturbation=perturbation, desc_dir=desc_dir, loss_x=loss, grad_x=gradient, nhl_rf=self.nhl_rf)

            next_step = desc_dir * atleast_kd(stepsize, desc_dir.ndim)
            if step < 1:
                norm_step = next_step.flatten(1).norms.lp(p=2, axis=1)
                norm_step = atleast_kd(norm_step, next_step.ndim)
                max_step = 5
                next_step = ep.where(
                    norm_step > max_step,
                    max_step * next_step / atleast_kd(norm_step, next_step.ndim),
                    next_step
                )

            advs += next_step
            advs = ep.clip(advs, *model.bounds)
            is_advs = self.is_adversarial(advs)
            found_advs = ep.logical_or(found_advs, is_advs)

            next_step_norm = next_step.raw.flatten(1).norm(dim=1)
            norms = ep.astensor((advs - x).raw.flatten(1).norm(dim=1))
            closer = norms < best_advs_norms
            new_best = ep.logical_and(closer, is_advs)

            best_advs = ep.where(atleast_kd(new_best, best_advs.ndim), advs, best_advs)

            # best_advs = ep.where(ep.astensor(found_advs),best_advs,delta+inputs_)
            best_advs_norms = ep.where(new_best, norms, best_advs_norms)

            self.c_sigma = ep.where(is_advs, self.c_sigma / self.c_sigma_gain, self.c_sigma * self.c_sigma_gain)
            if self.abort_early and ep.all(found_advs) and step > self.min_steps:
                break

            if self.max_queries is not None and all(self.queries > self.max_queries):
                break

            true_norm_grad = gradient.flatten(1).norms.lp(p=2, axis=1)
            norm_grad = self.get_norm_grad(gradient)
            logits_gt, logits_p = self.get_logits(advs)
            for i in range(len(x)):
                self.history_norms[i].append(float(norms[i].raw.cpu()))
                self.armijo_step[i].append(float(stepsize[i].raw.cpu()))
                self.is_advs_history[i].append(is_advs[i].raw.cpu())
                self.perturbation_added_dist[i].append(next_step_norm[i].cpu())
                self.loss_gt_history[i].append(logits_gt[i].raw.cpu())
                self.loss_p_history[i].append(logits_p[i].raw.cpu())
                self.query_history[i].append(self.queries[i] - it_query_start[i])
                self.gradient_norm[i].append(norm_grad[i].raw.cpu())
                self.true_gradient_norm[i].append(true_norm_grad[i].raw.cpu())
                self.descent_dir_norm[i].append(desc_dir_norm[i].raw.cpu())

        return restore_type(best_advs)

    def get_logits(self, x: ep.Tensor) -> Tuple[ep.Tensor, ep.Tensor]:

        """
        The loss used to study adverariality. If <0, then the image is adversarial
        """
        row = len(self.classes)
        rows = range(row)
        self.queries += 1
        mod = self.model(x)
        logits = ep.softmax(mod)
        return logits[rows, self.classes], logits[rows, best_other_classes(logits, self.classes)]
    
    def get_norm_grad(self, grad):
        norm_grad = grad.flatten(1).norms.lp(p=2, axis=1)
        # norm_grad = ep.where(norm_grad <= 10e-4, 10e-4, norm_grad)
        # norm_grad = ep.where(norm_grad >= 2, 2, norm_grad)
        return norm_grad

    def _d2(self, grad, perturbation, norm_grad):
        inner_product = (grad * perturbation).flatten(1).sum(axis=1)
        inner_product /=  (norm_grad**2)
        d2 = atleast_kd(inner_product, grad.ndim) * grad
        d2 -= perturbation
        return d2

    def _d1(self, grad, loss_x, norm_grad):
        d1 = - atleast_kd(loss_x / norm_grad**2, grad.ndim) * grad
        return d1

    def descent_dir(self, perturbation: ep.Tensor, grad: ep.Tensor, loss_x: ep.Tensor) -> ep.Tensor:
        norm_grad = self.get_norm_grad(grad)
        d2 = self._d2(grad, perturbation=perturbation, norm_grad=norm_grad)
        d1 = self._d1(grad, loss_x=loss_x, norm_grad=norm_grad)

        d2_norm = d2.flatten(1).norms.lp(p=2, axis=1)
        d1_norm = d1.flatten(1).norms.lp(p=2, axis=1)

        for i in range(len(perturbation)):
            self.d1_norm[i].append(float(d1_norm[i].raw.cpu()))
            self.d2_norm[i].append(float(d2_norm[i].raw.cpu()))

        d1_normalized = d1 / atleast_kd(d1_norm, d1.ndim)
        d2_normalized = d2 / atleast_kd(d2_norm, d2.ndim)
        norm_perturbation = perturbation.flatten(1).norms.lp(p=2, axis=1)
        d2 = ep.where(
            atleast_kd(d2_norm > self.d1_d2_ratio * norm_perturbation, d2.ndim),
            self.d1_d2_ratio * d2_normalized * atleast_kd(norm_perturbation, d1.ndim),
            d2
        )
        
        # d2 = ep.where(
        #     atleast_kd(ep.logical_and(
        #         d2_norm > self.d1_d2_ratio * d1_norm,
        #         loss_x > 0
        #     ), d2.ndim),
        #     self.d1_d2_ratio * d2_normalized * atleast_kd(d1_norm, d1.ndim),
        #     d2
        # )
        d2_norm = d2.flatten(1).norms.lp(p=2, axis=1)
        for i in range(len(perturbation)):
            self.d2_kept_norm[i].append(float(d2_norm[i].raw.cpu()))
        d = d1 + d2
        return d 

    # find the best stepsize thanks to the armijo's rule
    def pred_stepsize(self, x, perturbation: ep.Tensor, desc_dir: ep.Tensor, loss_x: ep.Tensor, grad_x: ep.Tensor, nhl_rf: bool = True) -> ep.Tensor:
        step_armijo = ep.full_like(loss_x, 0)
        norm_perturbation = perturbation.flatten(1).norms.lp(p=2, axis=1)
        norm_grad = self.get_norm_grad(grad_x)

        # sigm = ep.maximum((self.smooth * norm_perturbation) / norm_grad, (self.sigma + 1))

        norm_perturbation_and_d = (perturbation + desc_dir).flatten(1).norms.lp(p=2, axis=1)

        sigm = (self.smooth * norm_perturbation) / norm_grad

        if nhl_rf:
            inner_product = (grad_x * perturbation).flatten(1).sum(axis=1)
            inner_product /=  (norm_grad**2)
            sigm = ep.where(
                abs(loss_x) >= self.delta,
                abs((1/loss_x)*inner_product),
                norm_perturbation / norm_grad
            )

        
        else:
            sigm = ep.where(
                abs(loss_x) >= self.delta * self.g_0,
                ep.maximum(
                    norm_perturbation / norm_grad,
                    0.5 * norm_perturbation_and_d**2 / abs(loss_x)
                ),
                norm_perturbation / norm_grad
            )
        sigm *= self.eta

        for i in range(len(x)):
            self.norm_grad_history[i].append(float(norm_grad[i].raw.cpu()))
            self.perturbation_grad_ratio_history[i].append(float(sigm[i].raw.cpu()))

        # sigm *= self.c_sigma

        self.sigma = sigm

        for i in range(len(x)):
            self.sigm_history[i].append(float(sigm[i].raw.cpu()))

        mer = merit(sigm, norm_perturbation, loss_x)
        grad_mer = grad_merit(sigm, perturbation, loss_x, grad_x)
        cond = (desc_dir * grad_mer).flatten(1).sum(axis=1) * self.omega
        found_armijo = ep.zeros_like(loss_x).bool()
        k = 0
        while not found_armijo.all() and k <= 4: #((self.tau**k) > 10e-4):
            x_step = perturbation + self.gamma_speed_step * desc_dir * self.tau**k
            # x_step = ep.clip(x_step, *self.model.bounds)

            _, loss_step = self.loss_fun(x_step + x)
            norm_step = x_step.flatten(1).norms.lp(p=2, axis=1)
            respect_armijo = merit(sigm, norm_step, loss_step) - mer <= cond * self.tau**k
            respect_armijo = respect_armijo.bool()

            respect_armijo = ep.logical_and(respect_armijo, loss_step > -0.2)

            respect_armijo = ep.where(
                self.loss_under,
                ep.logical_and(respect_armijo, loss_step < 0.2),
                respect_armijo)


            alr_armijo = ep.where(respect_armijo, self.tau**k, 0)
            for i, e in enumerate(step_armijo):
                if e != 0:
                    self.queries[i] -= 1
            step_armijo = ep.where(step_armijo == 0, step_armijo + alr_armijo, step_armijo)
            found_armijo = step_armijo != 0
            k += 1

        respect_armijo = step_armijo != 0
        for i in range(len(x)):
            self.armijo_respect[i].append(float(respect_armijo[i].raw.cpu()))

        step_armijo = ep.where(step_armijo == 0, self.tau ** (k - 1), step_armijo)
        return step_armijo
        

def best_other_classes(logits: ep.Tensor, exclude: ep.Tensor) -> ep.Tensor:
    other_logits = logits - ep.onehot_like(logits, exclude, value=np.inf)
    return other_logits.argmax(axis=1)


# define the merit function and her gradient for the armijo's rule
def merit(sigma: ep.Tensor, norm_x: ep.Tensor, loss_x: ep.Tensor) -> ep.Tensor:
    # mer = ep.where(
    #         loss_x > 0,
    #         ep.abs(loss_x),
    #         0.5 * (norm_x) ** 2 + sigma * ep.abs(loss_x)
    #     )

    # mer = ep.where(
    #     loss_x > 0,
    #     ep.abs(loss_x),
    #     (norm_x) ** 2
    # )
    mer  = 0.5 * (norm_x) ** 2 + sigma * ep.abs(loss_x)
    return mer


def grad_merit(
    sigma: ep.Tensor, x: ep.Tensor, loss_x: ep.Tensor, grad_x: ep.Tensor
) -> ep.Tensor:
    sigma = atleast_kd(sigma, grad_x.ndim)
    sign = atleast_kd(ep.sign(loss_x), grad_x.ndim)
    return x + sigma * grad_x * sign
