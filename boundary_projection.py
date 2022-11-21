import torch
import torch.nn as nn
import numpy as np

class BP:
    def __init__(self,
                steps=50,
                gamma = 0.3,
                num_classes=1000,
                constraint=None,
                device = torch.device('cpu')):
        self.steps = steps
        self.gamma = gamma
        self.device = device
        self.num_classes = num_classes
        self.constraint = constraint
        self.queries = None

    def gamma_step(self, current_step):
        rate = current_step/(self.steps+1.0)
        rage = 1-self.gamma
        epsi = self.gamma + rate*rage
        return epsi

    def classif_loss(self, model, inputs_variable ,labels):
        if self.constraint is not None:
            perturbation = inputs_variable - self.inputs
            perturbation = self.constraint(perturbation)
            inputs_variable = self.inputs + perturbation
            inputs_variable = torch.autograd.Variable(inputs_variable, requires_grad=True)

        self.queries += 1
        prediction = model(inputs_variable / 255)
        labels_onehot = torch.zeros_like(prediction)
        labels_onehot.scatter_(1, labels.unsqueeze(1).long(),1)

        adversarial_loss = prediction*labels_onehot

        adversarial_loss = adversarial_loss.sum(axis=1)
        self.queries += 1
        adversarial_loss.backward(torch.ones_like(adversarial_loss), retain_graph=True)
        gradients = inputs_variable.grad.data
        return(prediction.argmax(1),gradients, adversarial_loss)

    def normalize(self, tensor_to_normalize):
        tensor_norm = tensor_to_normalize.flatten(1).norm(dim=1).view(self.batch_size,1,1,1)
        #Prevent inf / nan errors
        tensor_norm[tensor_norm==0]=1
        normalized_tensor = tensor_to_normalize/tensor_norm
        return(normalized_tensor, tensor_norm)

    def calculate_factor(self, loss, grad_norm,ratio):
        steps_ratio = self.steps*ratio
        decreasing_term = 1/(steps_ratio*self.gamma+(steps_ratio*(steps_ratio-1))/2*(1-self.gamma)/(self.steps+1))
        return(loss.view(self.batch_size,1,1,1)/grad_norm*decreasing_term)

    def update_best(self, current_adversarial, is_adversarial ,best_adversarial, best_norm):

        return(best_adv, best_norm)

    def _is_adversarial(self, model, x, y):
        p = model(x / 255).argmax(1)
        return (p!=y).view(self.batch_size,1,1,1)

        u = x - self.inputs
        is_advs = []
        #for alpha in np.linspace(1, 5, 20):
        #    is_advs.append((model((self.inputs + alpha * u).clip(0, 255) / 255).argmax(1) != y).unsqueeze(0))

        for _ in range(20):
            is_advs.append((model((x / 255 + torch.randn_like(x) * 0.1).clip(0, 1)).argmax(1) != y).unsqueeze(0))
        
        is_advs = torch.cat(is_advs, dim=0)
        is_advs = torch.all(is_advs, 0)
        return is_advs.view(self.batch_size,1,1,1)


    def __call__(self,model,inputs,labels,targeted=False, epsilons=None):
        self.queries = np.zeros(len(inputs))
        self.inputs = inputs
        batch_size = inputs.shape[0]
        self.batch_size = batch_size
        multiplier = 1 if targeted else -1
        best_adv = inputs.clone()
        best_norm = (torch.ones(batch_size,1,1,1)*1e6).to(best_adv.device)
        adv = inputs.clone()
        adv = torch.autograd.Variable(adv, requires_grad=True)
        pred_labels, grad, adversarial_loss = self.classif_loss(model, adv, labels)
        normalized_grad, grad_norm = self.normalize(grad)

        #Approximate the factor required to end Stage 1 within a ratio of the total steps
        ratio_factor =  self.calculate_factor(adversarial_loss, grad_norm, ratio=0.2)
        is_adv = (pred_labels!=labels).view(batch_size,1,1,1)
        already_adversarial = is_adv
        i=0
        ever_found = already_adversarial #help control stage 1 in case an image is already adversarial
        while i<self.steps:
            #Update Gamma
            gammas = self.gamma_step(i)

            #Stage 1: finding an adversarial sample quickly
            adv_stage_1 = adv + ratio_factor*multiplier*gammas*normalized_grad
            adv_stage_1 = torch.clamp(adv_stage_1,0,255)

            #Stage 2
            perturbation = adv-inputs

            normalized_perturbation, perturbation_norm = self.normalize(perturbation)
            #Projection of the perturbation onto the gradient vector
            proj_perturbation = (normalized_perturbation*-normalized_grad).flatten(1).sum(1).view(batch_size,1,1,1)

            #Case OUT
            # samples that are still adversarial  -> decrease distortion
            epsilons_out = gammas*perturbation_norm
            v_star = inputs + multiplier*proj_perturbation*normalized_grad
            v_adv_diff = (adv-v_star)
            diff_normed, diff_norm = self.normalize(v_adv_diff)
            distortion_control_out = epsilons_out**2-proj_perturbation**2
            distortion_control_out = torch.max(torch.zeros(distortion_control_out.shape).to(best_adv.device), distortion_control_out)
            adv_stage_2_out = v_star + diff_normed*(distortion_control_out**0.5)
            #Case IN
            # samples that are not adversarial anymore -> increase distortion
            epsilons_in = perturbation_norm/gammas
            distortion_control_in = epsilons_in**2-perturbation_norm**2+proj_perturbation**2
            adv_stage_2_in = adv + multiplier*(proj_perturbation+distortion_control_in**0.5)*normalized_grad
            #Keep the right adversarial w.r.t. IN/OUT case
            adv_stage_2 = adv_stage_2_out*is_adv+adv_stage_2_in*~is_adv

            #If stage 1 has ever been succesful, keep stage 2. Keep stage 1 otherwise
            new_adv = adv_stage_1*(ever_found==0) + adv_stage_2*(ever_found!=0)
            new_adv = torch.clamp(new_adv,0,255)


            #Update grads, current adversarial samples
            adv = torch.autograd.Variable(new_adv, requires_grad=True)
            pred_labels, grad,_ = self.classif_loss(model, adv, labels)
            normalized_grad, grad_norm = self.normalize(grad)
            is_adv = self._is_adversarial(model, adv, labels)
            ever_found = ever_found+is_adv.float()

            better_norm = perturbation_norm<=best_norm
            better_norm_and_adv = better_norm*is_adv
            best_adv = adv*better_norm_and_adv + best_adv*~better_norm_and_adv
            best_norm = best_adv.view(batch_size,-1).norm(dim=1).view(batch_size,1,1,1)
            """

            perturbation_norm = (adv - inputs).view(self.batch_size,-1).norm(dim=1).view(self.batch_size,1,1,1)
            best_norm_me = ((adv - inputs) / 255).view(batch_size,-1).norm(dim=1).view(batch_size,1,1,1)
            #print("sqdfsdf", best_norm_me.cpu())
            #print(is_adv)
            better_norm = torch.logical_or(best_norm == 0, perturbation_norm<=best_norm)
            better_norm_and_adv = better_norm*is_adv
            best_adv = adv*better_norm_and_adv + best_adv*~better_norm_and_adv
            best_norm = (best_adv - inputs).view(batch_size,-1).norm(dim=1).view(batch_size,1,1,1)

            """
            i+=1
        best_adv = best_adv*(~already_adversarial.view(batch_size,1,1,1))+inputs*(already_adversarial.view(batch_size,1,1,1))
        return best_adv.detach(),None,None
