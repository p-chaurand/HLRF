from typing import Any, Optional
import numpy as np
import eagerpy as ep
import copy
import torch



def atleast_kd(x, k: int):
    shape = x.shape + (1,) * (k - x.ndim)
    return x.reshape(shape)

class iHL_RFAttack:
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
        max_queries=None
    ):
        self.steps = steps
        self.confidence = confidence
        self.tau = tau
        self.omega = omega
        self.smooth = smooth
        self.abort_early = abort_early
        self.min_steps = min_steps
        self.max_queries = max_queries
        
        self.queries = None
        self.gamma_speed_step = 1
        self.c_sigma_gain = c_sigma_gain
        self.targeted = False
        self.c_sigma_start = c_sigma_start

    def classif_loss(self, x):
        self.queries += 1
        prediction = self.model(x)
        labels_onehot = torch.zeros_like(prediction)
        labels_onehot.scatter_(1, self.classes.unsqueeze(1).long(),1)

        adversarial_loss = prediction*labels_onehot
        
        adversarial_loss = adversarial_loss.sum(axis=1)
        self.queries += 1
        adversarial_loss.backward(torch.ones_like(adversarial_loss), retain_graph=True)
        gradients = x.grad.data
        return (prediction.argmax(1),gradients, adversarial_loss, prediction)

    def loss_fun(self, x):

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

    def is_adversarial(self, x):
        if not (self.targeted):
            return self.model(x).argmax(axis=1) != self.classes
        else:
            return self.model(x).argmax(axis=1) == self.classes

    def run(
        self,
        model,
        x,
        classes,
        *,
        early_stop: Optional[float] = None,
        **kwargs: Any,
    ):
        self.queries = np.zeros(len(x))
        del inputs, criterion, kwargs

        self.targeted = False
        classes = classes

        assert len(classes) == len(x)


        self.model = model
        self.classes = classes

        loss_aux_and_grad = ep.value_and_grad_fn(x, self.loss_fun, has_aux=True)

        best_advs = copy.deepcopy(x)
        best_advs_norms = torch.ones_like(classes) * np.inf

        advs = copy.deepcopy(x)
        found_advs = torch.zeros_like(classes)
        self.c_sigma = torch.ones_like(classes) * self.c_sigma_start

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
        for step in range(self.steps):
            it_query_start =self.queries
            self.queries += 1
            _, loss, gradient = loss_aux_and_grad(advs)
            perturbation = advs - x
            desc_dir = self.descent_dir(perturbation, gradient, loss)
            stepsize = self.pred_stepsize(x=x, perturbation=perturbation, desc_dir=desc_dir, loss_x=loss, grad_x=gradient)

            next_step = desc_dir * atleast_kd(stepsize, desc_dir.ndim)
            advs += next_step
            advs = advs.clip(0, 1)
            is_advs = self.is_adversarial(advs)
            found_advs = torch.logical_or(found_advs, is_advs)

            next_step_norm = next_step.flatten(1).norm(dim=1)
            norms = (advs - x).flatten(1).norm(dim=1)
            closer = norms < best_advs_norms
            new_best = torch.logical_and(closer, is_advs)

            best_advs = torch.where(atleast_kd(new_best, best_advs.ndim), advs, best_advs)

            # best_advs = ep.where(ep.astensor(found_advs),best_advs,delta+inputs_)
            best_advs_norms = torch.where(new_best, norms, best_advs_norms)

            self.c_sigma = torch.where(is_advs, self.c_sigma / self.c_sigma_gain, self.c_sigma * self.c_sigma_gain)
            if self.abort_early and found_advs.all() and step > self.min_steps:
                break

            if self.max_queries is not None and all(self.queries > self.max_queries):
                break

            logits_gt, logits_p = self.get_logits(advs)
            for i in range(len(x)):
                self.history_norms[i].append(float(norms[i].cpu()))
                self.armijo_step[i].append(float(stepsize[i].cpu()))
                self.is_advs_history[i].append(is_advs[i].cpu())
                self.perturbation_added_dist[i].append(next_step_norm[i].cpu())
                self.loss_gt_history[i].append(logits_gt[i].cpu())
                self.loss_p_history[i].append(logits_p[i].cpu())
                self.query_history[i].append(self.queries[i] - it_query_start[i])

        return best_advs

    def get_logits(self, x):

        """
        The loss used to study adverariality. If <0, then the image is adversarial
        """
        row = len(self.classes)
        rows = range(row)
        self.queries += 1
        mod = self.model(x)
        logits = torch.nn.functional.softmax(mod)
        return logits[rows, self.classes], logits[rows, best_other_classes(logits, self.classes)]

    def descent_dir(self, perturbation, grad, loss_x):
        norm_grad = grad.flatten(1).norms(p=2, axis=1)
        norm_grad = torch.where(norm_grad == torch.zeros_like(norm_grad), torch.ones_like(norm_grad), norm_grad)
        gamma = ((grad * perturbation).flatten(1).sum(axis=1) - loss_x) / (norm_grad**2)
        res = atleast_kd(gamma, grad.ndim) * grad - perturbation
        return res

    # find the best stepsize thanks to the armijo's rule
    def pred_stepsize(self, x, perturbation, desc_dir, loss_x, grad_x):
        step_armijo = torch.full_like(loss_x, 0)
        norm_perturbation = perturbation.flatten(1).norms.lp(p=2, axis=1)
        grad_x_norm = grad_x.flatten(1).norms.lp(p=2, axis=-1)
        norm_grad = torch.where(grad_x_norm == 0, 1, grad_x_norm)  # assure that we dont divide by 0

        # sigm = ep.maximum((self.smooth * norm_perturbation) / norm_grad, (self.sigma + 1))
        sigm = (self.smooth * norm_perturbation) / norm_grad

        for i in range(len(x)):
            self.norm_grad_history[i].append(float(norm_grad[i].cpu()))
            self.perturbation_grad_ratio_history[i].append(float(sigm[i].cpu()))

        sigm *= self.c_sigma

        self.sigma = sigm

        for i in range(len(x)):
            self.sigm_history[i].append(float(sigm[i].cpu()))

        mer = merit(sigm, norm_perturbation, loss_x)
        grad_mer = grad_merit(sigm, perturbation, loss_x, grad_x)
        cond = (desc_dir * grad_mer).flatten(1).sum(axis=1) * self.omega
        found_armijo = torch.zeros_like(loss_x).bool()
        k = 0
        while not found_armijo.all() and ((self.tau**k) > 10e-4):
            x_step = perturbation + self.gamma_speed_step * desc_dir * self.tau**k
            # x_step = ep.clip(x_step + inputs_, *model.bounds) - inputs_

            _, loss_step = self.loss_fun(x_step + x)
            norm_step = x_step.flatten(1).norms.lp(p=2, axis=1)
            respect_armijo = merit(sigm, norm_step, loss_step) - mer <= cond * self.tau**k
            respect_armijo = respect_armijo.bool()

            alr_armijo = torch.where(respect_armijo, self.tau**k, 0)
            for i, e in enumerate(step_armijo):
                if e != 0:
                    self.queries[i] -= 1
            step_armijo = torch.where(step_armijo == 0, step_armijo + alr_armijo, step_armijo)
            found_armijo = step_armijo != 0
            k += 1

        respect_armijo = step_armijo != 0
        for i in range(len(x)):
            self.armijo_respect[i].append(float(respect_armijo[i].raw.cpu()))

        step_armijo = torch.where(step_armijo == 0, self.tau ** (k - 1), step_armijo)
        return step_armijo


def best_other_classes(logits, exclude):
    other_logits = logits - torch.onehot_like(logits, exclude, value=np.inf)
    return other_logits.argmax(axis=1)


# define the merit function and her gradient for the armijo's rule
def merit(sigma, norm_x, loss_x):
    return 0.5 * (norm_x) ** 2 + sigma * torch.abs(loss_x)


def grad_merit(sigma, x, loss_x, grad_x):
    sigma = atleast_kd(sigma, grad_x.ndim)
    sign = atleast_kd(torch.sign(loss_x), grad_x.ndim)
    return x + sigma * grad_x * sign
