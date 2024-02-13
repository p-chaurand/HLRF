from typing import Union, Tuple, Any
from .base import MinimizationAttack
import eagerpy as ep

from ..distances import l2
from ..devutils import atleast_kd
from ..devutils import flatten

from ..models import Model

from ..criteria import Misclassification
from ..criteria import TargetedMisclassification

from .base import T
from .base import get_criterion
from .base import raise_if_kwargs
from .base import verify_input_bounds



class iHL_RFAttack(MinimizationAttack):
    """Implementation of the improved Hasofer-Lind, Rackwitz-Fiessler Attack.

    Args:
        steps : Number of optimization steps.
        func_grad_hl : Function to give the proportion (between 0 and 1) between the gradient descent direction and the HL-RF direction (default is linear)
            This function should take the actual step iteration and the starting proportion as input and return a couple (grad_proportion, hl_proportion).
        beta : Starting proportion between the gradient descent direction and the HL-RF direction.
        confidence : Confidence required for an example to be marked as adversarial (just considered in the loss to go behind the frontier).
            Controls the gap between example and decision boundary.
        abort_early : Stop inner search as soons as an adversarial example has been found.
        min_steps : if abort early dont stop before a minimum step.


    """

    distance = l2

    def __init__(
        self,
        steps: int = 50,
        func_grad_hl = None,
        alpha: float = 0.9,
        beta: float = 0.5,
        confidence: float = 0,
        abort_early: bool = False,
        min_steps: int = 25,
    ):
        self.steps = steps
        self.confidence = confidence
        self.abort_early = abort_early
        self.min_steps = min_steps
        self.alpha = alpha
        self.beta = beta


        if func_grad_hl is None:
            self.func_grad_hl = lambda k : self.func_grad_hl_default(k)
        else:
            self.func_grad_hl = lambda k : func_grad_hl(k)

    def func_grad_hl_default(self, k):
        # Linear function that goes from beta to 1 for k from 0 to alpha*steps and 1 after
        if self.beta == 0:
            prop = k/(self.alpha*self.steps)
        else:    
            prop = self.beta + (1-self.beta) * (k / (self.alpha*self.steps))
        prop = min(1,prop)
        prop = max(0,prop)
        # desc_dir, hl_rf_dir
        return (1-prop,prop)

    def loss_fun(self, image: ep.Tensor) -> Tuple[ep.Tensor, Tuple[ep.Tensor,ep.Tensor]]:
        """
        The loss used to study adverariality. If <0, then the image is adversarial
        """
        labels = self.labels
        model = self.model
        row = len(labels)
        rows = range(row)
        mod = model(image)
        logits = ep.softmax(mod)

        if self.targeted:
            c_minimize = self.best_other_classes(logits, labels)
            c_maximize = labels  # target_classes
        else:
            c_minimize = labels  # labels
            c_maximize = self.best_other_classes(logits, labels)

        loss = (logits[rows, c_minimize] - logits[rows, c_maximize]) + self.confidence
        return loss.sum(), (loss,logits)
    
    def best_other_classes(self, logits: ep.Tensor, exclude: ep.Tensor) -> ep.Tensor:
        other_logits = logits - ep.onehot_like(logits, exclude, value=ep.inf)
        return other_logits.argmax(axis=-1)

    def run(self,
            model: Model,
            inputs_: T,
            criterion: Union[Misclassification, TargetedMisclassification, T],
            early_stop: bool = False,
            **kwargs: Any,
            ) -> T:

        raise_if_kwargs(kwargs)
        input, restore_type = ep.astensor_(inputs_)
        criterion_ = get_criterion(criterion)
        del inputs_, criterion, kwargs
        verify_input_bounds(input, model)

        if isinstance(criterion_, Misclassification):
            self.targeted = False
            #labels = ep.argmax(model(input),axis=1)
            labels = criterion_.labels

        elif isinstance(criterion_, TargetedMisclassification):
            self.targeted = True
            labels = criterion_.target_classes


        assert len(labels) == len(input)

        self.model = model
        self.labels = labels

        loss_aux_and_grad = ep.value_and_grad_fn((input), self.loss_fun, has_aux=True)


        best_advs = input
        best_advs_norms = ep.ones_like(labels) * ep.inf


        advs = input
        perturbation = ep.zeros_like(advs)
        _, (loss,pred_labels), gradient = loss_aux_and_grad(advs)


        found_advs = criterion_(self.labels,pred_labels)
        already_advs = found_advs


        for step in range(self.steps):
            desc_dir = self.descent_dir(perturbation, gradient, loss)
            hl_rf_dir = self.hl_rf_dir(perturbation, gradient, loss)

            perturbation = (self.func_grad_hl(step)[0]) * desc_dir + (self.func_grad_hl(step)[1]) * hl_rf_dir

            advs = input + perturbation
            
            advs = ep.clip(advs, *model.bounds)

            perturbation = advs - input

            _, (loss,pred_labels), gradient = loss_aux_and_grad(advs)

            is_advs = criterion_(self.labels,pred_labels)
            found_advs = ep.logical_or(found_advs, is_advs)

            norms = (advs * perturbation).flatten(1).sum(axis=1)

            closer = norms < best_advs_norms
            new_best = ep.logical_and(closer, is_advs)

            best_advs = ep.where( atleast_kd(new_best, best_advs.ndim), advs, best_advs)

            best_advs_norms = ep.where(new_best, norms, best_advs_norms)

            
            if self.abort_early and found_advs.all() and step > self.min_steps:
                break
        
        best_advs = ep.clip(best_advs, *model.bounds)
        best_advs = ep.where(atleast_kd(already_advs,input.ndim), input, best_advs)
        return restore_type(best_advs)


    def descent_dir(self, perturbation, grad, loss_x):
        norm_grad = grad.flatten(1).norms.lp(p=2, axis=1)
        norm_grad = ep.maximum(norm_grad, 1e-12)
        step_size = loss_x / (norm_grad**2)
        gamma = atleast_kd(step_size, grad.ndim) * grad
        return perturbation - gamma


    def hl_rf_dir(self, perturbation, grad, loss_x):
        norm_grad = grad.flatten(1).norms.lp(p=2, axis=1)
        norm_grad = ep.maximum(norm_grad, 1e-12)
        gamma = (grad * perturbation).flatten(1).sum(axis=1) - loss_x
        return atleast_kd(gamma / (norm_grad**2),grad.ndim) * grad