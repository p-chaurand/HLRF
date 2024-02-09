from typing import Union, Tuple, Any, Optional
from ..criteria import Misclassification
from ..criteria import TargetedMisclassification
from .base import MinimizationAttack
from ..models import Model
import eagerpy as ep
from ..devutils import flatten
from ..devutils import atleast_kd
from ..distances import l2

from .base import T
from .base import get_criterion
from .base import raise_if_kwargs
from .base import verify_input_bounds


class BP(MinimizationAttack):

	distance = l2

	def __init__(self,
				steps: int = 50,
				gamma: float = 0.3,
				num_classes: int =1000,
				targeted: bool = False,
				confidence: float = 0.1,):
		self.steps = steps
		self.gamma = gamma
		self.num_classes = num_classes
		self.query_count: int = 0
		self.targeted = targeted
		self.confidence = confidence
		

	def gamma_step(self, current_step):
		rate = current_step/(self.steps+1.0)
		rage = 1-self.gamma
		epsi = self.gamma + rate*rage
		return epsi

	def classif_loss(self,
					image: ep.Tensor,
					labels: ep.Tensor,
					model: Model,
		) -> Tuple[ep.Tensor, Tuple[ep.Tensor,ep.Tensor]]:

			"""
			The loss used to study adverariality. If <0, then the image is adversarial
			"""
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

			loss = (
				logits[rows, c_minimize] - logits[rows, c_maximize]
			) + self.confidence

			return loss.sum(), (loss,ep.argmax(logits,axis=-1))

	def best_other_classes(self,logits: ep.Tensor, exclude: ep.Tensor) -> ep.Tensor:
		other_logits = logits - ep.onehot_like(logits, exclude, value=10e6)
		res = other_logits.argmax(axis=-1)

		return res

	def normalize(self, tensor_to_normalize):
		tensor_norm = l2(tensor_to_normalize,ep.zeros_like(tensor_to_normalize))
		#Prevent inf / nan errors
		tensor_norm = ep.where(tensor_norm==0,ep.ones_like(tensor_norm),tensor_norm)
		tensor_norm = atleast_kd(tensor_norm, tensor_to_normalize.ndim)
		normalized_tensor = tensor_to_normalize/tensor_norm
		return(normalized_tensor, tensor_norm)
	
	def calculate_factor(self, loss, grad_norm,ratio):
		loss = atleast_kd(loss, grad_norm.ndim)
		steps_ratio = self.steps*ratio
		decreasing_term = 1/(steps_ratio*self.gamma+(steps_ratio*(steps_ratio-1))/2*(1-self.gamma)/(self.steps+1))
		return(loss/(grad_norm*decreasing_term))



	def run(self,
			model: Model,
			inputs_: T,
			criterion: Union[Misclassification, TargetedMisclassification, T],
			early_stop: bool = False,
			**kwargs: Any,
			) -> T:

		raise_if_kwargs(kwargs)
		inputs, restore_type = ep.astensor_(inputs_)
		criterion_ = get_criterion(criterion)
		del inputs_, criterion, kwargs
		verify_input_bounds(inputs, model)

		N = len(inputs)

		if isinstance(criterion_, Misclassification):
			targeted = False
			labels = criterion_.labels

		elif isinstance(criterion_, TargetedMisclassification):
			targeted = True
			labels = criterion_.target_classes

		loss_aux_and_grad = ep.value_and_grad_fn(ep.zeros_like(inputs), self.classif_loss,has_aux=True)
		
		batch_size = inputs.shape[0]
		self.batch_size = batch_size
		multiplier = 1 if targeted else -1
		best_adv = inputs
		best_norm = (ep.ones(inputs,(batch_size,1,1,1))*ep.inf)
		adv = inputs
		_ , (adversarial_loss,pred_labels),grad = loss_aux_and_grad(adv, labels,model)
		normalized_grad, grad_norm = self.normalize(grad)

		#Approximate the factor required to end Stage 1 within a ratio of the total steps
		ratio_factor =  self.calculate_factor(adversarial_loss, grad_norm, ratio=0.2)
		is_adv = (pred_labels!=labels)
		already_adversarial = is_adv
		i=0
		ever_found = already_adversarial #help control stage 1 in case an image is already adversarial
		while i<self.steps:
			#Update Gamma
			gammas = self.gamma_step(i)

			#Stage 1: finding an adversarial sample quickly
			adv_stage_1 = adv + ratio_factor*multiplier*gammas*normalized_grad

			#Stage 2
			perturbation = adv-inputs
			normalized_perturbation, perturbation_norm = self.normalize(perturbation)
			#Projection of the perturbation onto the gradient vector
			proj_perturbation = (normalized_perturbation*-normalized_grad)

			#Case OUT
			# samples that are still adversarial  -> decrease distortion
			epsilons_out = gammas*perturbation_norm
			v_star = inputs + multiplier*proj_perturbation*normalized_grad
			v_adv_diff = (adv-v_star)
			diff_normed, diff_norm = self.normalize(v_adv_diff)
			distortion_control_out = epsilons_out**2-proj_perturbation**2
			distortion_control_out = ep.maximum(ep.zeros_like(distortion_control_out), distortion_control_out)
			adv_stage_2_out = v_star + diff_normed*(distortion_control_out**0.5)


			#Case IN
			# samples that are not adversarial anymore -> increase distortion
			epsilons_in = perturbation_norm/gammas
			distortion_control_in = epsilons_in**2-perturbation_norm**2+proj_perturbation**2
			adv_stage_2_in = adv + multiplier*(proj_perturbation+distortion_control_in**0.5)*normalized_grad
			#Keep the right adversarial w.r.t. IN/OUT case
			adv_stage_2 = ep.where(atleast_kd(is_adv,adv.ndim),adv_stage_2_out,adv_stage_2_in)

			#If stage 1 has ever been succesful, keep stage 2. Keep stage 1 otherwise
			new_adv = ep.where(atleast_kd(ever_found,adv.ndim),adv_stage_2,adv_stage_1)
			#Update grads, current adversarial samples
			adv = new_adv
			_ , (adversarial_loss,pred_labels),grad = loss_aux_and_grad(adv, labels, model)
			normalized_grad, grad_norm = self.normalize(grad)
			is_adv = (pred_labels!=labels)
			adv_norm = l2(adv,inputs)
			adv_norm = atleast_kd(adv_norm, adv.ndim)
			ever_found = ep.logical_or(ever_found,is_adv)
			


			better_norm = adv_norm<=best_norm

			better_norm_and_adv = ep.logical_and(better_norm,atleast_kd(is_adv,adv.ndim))
			best_adv = ep.where(better_norm_and_adv,adv,best_adv)
			best_norm = l2(best_adv, inputs)
			best_norm = atleast_kd(best_norm, best_adv.ndim)


			i+=1

		already_adversarial = atleast_kd(already_adversarial,inputs.ndim)
		best_adv = ep.where(already_adversarial,inputs,best_adv)
		return restore_type(best_adv)
