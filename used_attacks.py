import torch
import torch.nn as nn
import time
import torch.nn.functional as F
import numpy as np
from scipy import stats as st
import random
from torch.autograd import Variable
from dct import *

class Attack(object):
    r"""
    Base class for all attacks.

    .. note::
        It automatically set device to the device where given model is.
        It basically changes training mode to eval during attack process.
        To change this, please see `set_training_mode`.
    """
    def __init__(self, name, model):
        r"""
        Initializes internal attack state.

        Arguments:
            name (str): name of attack.
            model (torch.nn.Module): model to attack.
        """

        self.attack = name
        self.model = model
        self.model_name = str(model).split("(")[0]
        self.device = next(model.parameters()).device

        self._attack_mode = 'default'
        self._targeted = False
        self._return_type = 'float'
        self._supported_mode = ['default']

        self._model_training = False
        self._batchnorm_training = False
        self._dropout_training = False

    def forward(self, *input):
        r"""
        It defines the computation performed at every call.
        Should be overridden by all subclasses.
        """
        raise NotImplementedError

    def get_mode(self):
        r"""
        Get attack mode.

        """
        return self._attack_mode

    def set_mode_default(self):
        r"""
        Set attack mode as default mode.

        """
        self._attack_mode = 'default'
        self._targeted = False
        print("Attack mode is changed to 'default.'")

    def set_mode_targeted_by_function(self, target_map_function=None):
        r"""
        Set attack mode as targeted.

        Arguments:
            target_map_function (function): Label mapping function.
                e.g. lambda images, labels:(labels+1)%10.
                None for using input labels as targeted labels. (Default)

        """
        if "targeted" not in self._supported_mode:
            raise ValueError("Targeted mode is not supported.")

        self._attack_mode = 'targeted'
        self._targeted = True
        self._target_map_function = target_map_function
        print("Attack mode is changed to 'targeted.'")

    def set_mode_targeted_least_likely(self, kth_min=1):
        r"""
        Set attack mode as targeted with least likely labels.
        Arguments:
            kth_min (str): label with the k-th smallest probability used as target labels. (Default: 1)

        """
        if "targeted" not in self._supported_mode:
            raise ValueError("Targeted mode is not supported.")

        self._attack_mode = "targeted(least-likely)"
        self._targeted = True
        assert (kth_min > 0)
        self._kth_min = kth_min
        self._target_map_function = self._get_least_likely_label
        print("Attack mode is changed to 'targeted(least-likely).'")

    def set_mode_targeted_random(self):
        r"""
        Set attack mode as targeted with random labels.
        Arguments:
            num_classses (str): number of classes.

        """
        if "targeted" not in self._supported_mode:
            raise ValueError("Targeted mode is not supported.")

        self._attack_mode = "targeted(random)"
        self._targeted = True
        self._target_map_function = self._get_random_target_label
        print("Attack mode is changed to 'targeted(random).'")

    def set_return_type(self, type):
        r"""
        Set the return type of adversarial images: `int` or `float`.

        Arguments:
            type (str): 'float' or 'int'. (Default: 'float')

        .. note::
            If 'int' is used for the return type, the file size of 
            adversarial images can be reduced (about 1/4 for CIFAR10).
            However, if the attack originally outputs float adversarial images
            (e.g. using small step-size than 1/255), it might reduce the attack
            success rate of the attack.

        """
        if type == 'float':
            self._return_type = 'float'
        elif type == 'int':
            self._return_type = 'int'
        else:
            raise ValueError(type + " is not a valid type. [Options: float, int]")

    def set_training_mode(self, model_training=False, batchnorm_training=False, dropout_training=False):
        r"""
        Set training mode during attack process.

        Arguments:
            model_training (bool): True for using training mode for the entire model during attack process.
            batchnorm_training (bool): True for using training mode for batchnorms during attack process.
            dropout_training (bool): True for using training mode for dropouts during attack process.

        .. note::
            For RNN-based models, we cannot calculate gradients with eval mode.
            Thus, it should be changed to the training mode during the attack.
        """
        self._model_training = model_training
        self._batchnorm_training = batchnorm_training
        self._dropout_training = dropout_training

    def save(self, data_loader, save_path=None, verbose=True, return_verbose=False, save_pred=False):
        r"""
        Save adversarial images as torch.tensor from given torch.utils.data.DataLoader.

        Arguments:
            save_path (str): save_path.
            data_loader (torch.utils.data.DataLoader): data loader.
            verbose (bool): True for displaying detailed information. (Default: True)
            return_verbose (bool): True for returning detailed information. (Default: False)
            save_pred (bool): True for saving predicted labels (Default: False)

        """
        if save_path is not None:
            image_list = []
            label_list = []
            if save_pred:
                pre_list = []

        correct = 0
        total = 0
        l2_distance = []

        total_batch = len(data_loader)

        given_training = self.model.training
        given_return_type = self._return_type
        self._return_type = 'float'

        for step, (images, labels) in enumerate(data_loader):
            start = time.time()
            adv_images = self.__call__(images, labels)

            batch_size = len(images)

            if verbose or return_verbose:
                with torch.no_grad():
                    if given_training:
                        self.model.eval()
                    outputs = self.model(adv_images)
                    _, pred = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    right_idx = (pred == labels.to(self.device))
                    correct += right_idx.sum()
                    end = time.time()
                    delta = (adv_images - images.to(self.device)).view(batch_size, -1)
                    l2_distance.append(torch.norm(delta[~right_idx], p=2, dim=1))

                    rob_acc = 100 * float(correct) / total
                    l2 = torch.cat(l2_distance).mean().item()
                    progress = (step+1)/total_batch*100
                    elapsed_time = end-start
                    if verbose:
                        self._save_print(progress, rob_acc, l2, elapsed_time, end='\r')

            if save_path is not None:
                if given_return_type == 'int':
                    adv_images = self._to_uint(adv_images.detach().cpu())
                    image_list.append(adv_images)
                else:
                    image_list.append(adv_images.detach().cpu())
                label_list.append(labels.detach().cpu())

                image_list_cat = torch.cat(image_list, 0)
                label_list_cat = torch.cat(label_list, 0)

                if save_pred:
                    pre_list.append(pred.detach().cpu())
                    pre_list_cat = torch.cat(pre_list, 0)
                    torch.save((image_list_cat, label_list_cat, pre_list_cat), save_path)
                else:
                    torch.save((image_list_cat, label_list_cat), save_path)

        # To avoid erasing the printed information.
        if verbose:
            self._save_print(progress, rob_acc, l2, elapsed_time, end='\n')

        if given_training:
            self.model.train()

        if return_verbose:
            return rob_acc, l2, elapsed_time

    def _save_print(self, progress, rob_acc, l2, elapsed_time, end):
        print('- Save progress: %2.2f %% / Robust accuracy: %2.2f %% / L2: %1.5f (%2.3f it/s) \t' \
              % (progress, rob_acc, l2, elapsed_time), end=end)

    @torch.no_grad()
    def _get_target_label(self, images, labels=None):
        r"""
        Function for changing the attack mode.
        Return input labels.
        """
        if self._targeted:
            given_training = self.model.training
            if given_training:
                self.model.eval()
            target_labels = self._target_map_function(images, labels)
            if given_training:
                self.model.train()
            return target_labels
        else:
            raise ValueError('Please define target_map_function.')

    @torch.no_grad()
    def _get_least_likely_label(self, images, labels=None):
        outputs = self.model(images)
        if labels is None:
            _, labels = torch.max(outputs, dim=1)
        n_classses = outputs.shape[-1]

        target_labels = torch.zeros_like(labels)
        for counter in range(labels.shape[0]):
            l = list(range(n_classses))
            l.remove(labels[counter])
            _, t = torch.kthvalue(outputs[counter][l], self._kth_min)
            target_labels[counter] = l[t]

        return target_labels.long().to(self.device)

    @torch.no_grad()
    def _get_random_target_label(self, images, labels=None):
        outputs = self.model(images)
        if labels is None:
            _, labels = torch.max(outputs, dim=1)
        n_classses = outputs.shape[-1]

        target_labels = torch.zeros_like(labels)
        for counter in range(labels.shape[0]):
            l = list(range(n_classses))
            l.remove(labels[counter])
            t = (len(l)*torch.rand([1])).long().to(self.device)
            target_labels[counter] = l[t]

        return target_labels.long().to(self.device)

    def _to_uint(self, images):
        r"""
        Function for changing the return type.
        Return images as int.
        """
        return (images*255).type(torch.uint8)

    def __str__(self):
        info = self.__dict__.copy()

        del_keys = ['model', 'attack']

        for key in info.keys():
            if key[0] == "_":
                del_keys.append(key)

        for key in del_keys:
            del info[key]

        info['attack_mode'] = self._attack_mode
        info['return_type'] = self._return_type

        return self.attack + "(" + ', '.join('{}={}'.format(key, val) for key, val in info.items()) + ")"

    def __call__(self, *input, **kwargs):
        given_training = self.model.training

        if self._model_training:
            self.model.train()
            for _, m in self.model.named_modules():
                if not self._batchnorm_training:
                    if 'BatchNorm' in m.__class__.__name__:
                        m = m.eval()
                if not self._dropout_training:
                    if 'Dropout' in m.__class__.__name__:
                        m = m.eval()
        else:
            self.model.eval()

        images = self.forward(*input, **kwargs)

        if given_training:
            self.model.train()

        if self._return_type == 'int':
            images = self._to_uint(images)

        return images


class IBTA_Logits_TI(Attack):

    def __init__(self, model, eps=8/255, alpha=2/255, steps=20, decay=1.0, kernel_name='gaussian',
                 len_kernel=5, nsig=3, resize_rate=0.9, diversity_prob=0.7, random_start=False,
                 loss=nn.CrossEntropyLoss(),targeted=False, target=None,normalize=True, gamma=0.1, lmd=0.1, sigma=0.1):
        super().__init__("IBTA_Logits_TI", model)
        self.eps = eps
        self.steps = steps
        self.decay = decay
        self.alpha = alpha
        self.resize_rate = resize_rate
        self.diversity_prob = diversity_prob
        self.random_start = random_start
        self.kernel_name = kernel_name
        self.len_kernel = len_kernel
        self.nsig = nsig
        self.stacked_kernel = torch.from_numpy(self.kernel_generation())
        self.loss = loss
        self.targeted = targeted
        self.target = target
        self.normalize = normalize
        self.gamma = gamma
        self.lmd = lmd
        self.sigma = sigma

    def norm(self, t):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        t[:, 0, :, :] = (t[:, 0, :, :] - mean[0])/std[0]
        t[:, 1, :, :] = (t[:, 1, :, :] - mean[1])/std[1]
        t[:, 2, :, :] = (t[:, 2, :, :] - mean[2])/std[2]

        return t
    
    def logit_loss(self, out, labels):
        return -torch.gather(out, 1, labels.view(-1, 1)).squeeze(-1).mean()

    def double_kl_div(self, p_output, q_output, get_softmax=True,gamma=0.01):

        KLDivLoss = nn.KLDivLoss(reduction='batchmean')
        
        return  gamma*KLDivLoss(F.log_softmax(p_output,dim=-1), F.softmax(q_output,dim=-1))+KLDivLoss(F.log_softmax(q_output,dim=-1), F.softmax(p_output,dim=-1))

    def forward(self, images, labels):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self.targeted:
            labels = self.target*torch.ones_like(labels)

        #loss = self.loss
        momentum = torch.zeros_like(images).detach().to(self.device)
        stacked_kernel = self.stacked_kernel.to(self.device)

        adv_delta = torch.zeros_like(images).to(self.device)
        
        clean_images = images.clone().detach()
        clean_images_shifted = clean_images+self.sigma*(torch.randn_like(clean_images))
        clean_images_shifted = torch.clamp(clean_images_shifted, min=0, max=1)

        if self.random_start:
            # Starting at a uniformly random point
            adv_delta = adv_delta + torch.empty_like(adv_delta).uniform_(-self.eps, self.eps)
            clean_images = torch.clamp(clean_images+adv_delta, min=0, max=1).detach()

        for _ in range(self.steps):
            #adv_images.requires_grad = True
            adv_delta.requires_grad = True
            adv_images = torch.clamp(clean_images+adv_delta, min=0, max=1)
            adv_images_shifted = torch.clamp(clean_images_shifted+adv_delta, min=0, max=1)

            if self.normalize:
                outputs = self.model(self.norm(self.input_diversity(adv_images)))
                outputs_shifted = self.model(self.norm(self.input_diversity(adv_images_shifted)))
            else:
                outputs = self.model(self.input_diversity(adv_images))
                outputs_shifted = self.model(self.input_diversity(adv_images_shifted))

            # Calculate loss
            if self.targeted:
                cost = -self.logit_loss(outputs, labels)-self.gamma*self.logit_loss(outputs_shifted, labels)-self.lmd*self.double_kl_div(outputs, outputs_shifted,gamma=self.gamma)
            else:
                cost = self.logit_loss(outputs, labels)+self.gamma*self.logit_loss(outputs_shifted, labels)-self.lmd*self.double_kl_div(outputs, outputs_shifted,gamma=self.gamma)
            # Update adversarial images
            grad = torch.autograd.grad(cost, adv_delta,
                                       retain_graph=False, create_graph=False)[0]
            # depth wise conv2d
            grad = F.conv2d(grad, stacked_kernel, stride=1, padding='same', groups=3)
            grad = grad / torch.mean(torch.abs(grad), dim=(1,2,3), keepdim=True)
            grad = grad + momentum*self.decay
            momentum = grad

            adv_delta = adv_delta.detach() + self.alpha*grad.sign()
            adv_delta = torch.clamp(adv_delta, min=-self.eps, max=self.eps)
        adv_images = torch.clamp(images + adv_delta, min=0, max=1).detach()

        return adv_images

    def kernel_generation(self):
        if self.kernel_name == 'gaussian':
            kernel = self.gkern(self.len_kernel, self.nsig).astype(np.float32)
        elif self.kernel_name == 'linear':
            kernel = self.lkern(self.len_kernel).astype(np.float32)
        elif self.kernel_name == 'uniform':
            kernel = self.ukern(self.len_kernel).astype(np.float32)
        else:
            raise NotImplementedError

        stack_kernel = np.stack([kernel, kernel, kernel])
        stack_kernel = np.expand_dims(stack_kernel, 1)
        return stack_kernel

    def gkern(self, kernlen=15, nsig=3):
        """Returns a 2D Gaussian kernel array."""
        x = np.linspace(-nsig, nsig, kernlen)
        kern1d = st.norm.pdf(x)
        kernel_raw = np.outer(kern1d, kern1d)
        kernel = kernel_raw / kernel_raw.sum()
        return kernel

    def ukern(self, kernlen=15):
        kernel = np.ones((kernlen,kernlen))* 1.0 /(kernlen*kernlen)
        return kernel

    def lkern(self, kernlen=15):
        kern1d = 1-np.abs(np.linspace((-kernlen+1)/2, (kernlen-1)/2, kernlen)/(kernlen+1)*2)
        kernel_raw = np.outer(kern1d, kern1d)
        kernel = kernel_raw / kernel_raw.sum()
        return kernel

    def input_diversity(self, x):
        img_size = x.shape[-1]
        img_resize = int(img_size * self.resize_rate)

        if self.resize_rate < 1:
            img_size = img_resize
            img_resize = x.shape[-1]

        rnd = torch.randint(low=img_size, high=img_resize, size=(1,), dtype=torch.int32)
        rescaled = F.interpolate(x, size=[rnd, rnd], mode='bilinear', align_corners=False)
        h_rem = img_resize - rnd
        w_rem = img_resize - rnd
        pad_top = torch.randint(low=0, high=h_rem.item(), size=(1,), dtype=torch.int32)
        pad_bottom = h_rem - pad_top
        pad_left = torch.randint(low=0, high=w_rem.item(), size=(1,), dtype=torch.int32)
        pad_right = w_rem - pad_left

        padded = F.pad(rescaled, [pad_left.item(), pad_right.item(), pad_top.item(), pad_bottom.item()], value=0)

        return padded if torch.rand(1) < self.diversity_prob else x



class IBTA_RAP_Logits_TI(Attack):

    def __init__(self, model, eps=8/255, alpha=2/255, steps=20, decay=1.0, kernel_name='gaussian',
                 len_kernel=5, nsig=3, resize_rate=0.9, diversity_prob=0.7, random_start=False,
                 loss=nn.CrossEntropyLoss(),targeted=False, target=None,normalize=True, k_ls = 100, epsilon_n = 12, gamma=0.1, lmd=0.1, sigma=0.1):
        super().__init__("IBTA_RAP_Logits_TI", model)
        self.eps = eps
        self.steps = steps
        self.decay = decay
        self.alpha = alpha
        self.resize_rate = resize_rate
        self.diversity_prob = diversity_prob
        self.random_start = random_start
        self.kernel_name = kernel_name
        self.len_kernel = len_kernel
        self.nsig = nsig
        self.stacked_kernel = torch.from_numpy(self.kernel_generation())
        self.loss = loss
        self.targeted = targeted
        self.target = target
        self.normalize = normalize
        self.k_ls = k_ls
        self.epsilon_n = epsilon_n
        self.gamma = gamma
        self.lmd = lmd
        self.sigma = sigma

    def norm(self, t):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        t[:, 0, :, :] = (t[:, 0, :, :] - mean[0])/std[0]
        t[:, 1, :, :] = (t[:, 1, :, :] - mean[1])/std[1]
        t[:, 2, :, :] = (t[:, 2, :, :] - mean[2])/std[2]

        return t
    
    def logit_loss(self, out, labels):
        return -torch.gather(out, 1, labels.view(-1, 1)).squeeze(-1).mean()

    def double_kl_div(self, p_output, q_output, get_softmax=True,gamma=0.01):

        KLDivLoss = nn.KLDivLoss(reduction='batchmean')
        
        return  gamma*KLDivLoss(F.log_softmax(p_output,dim=-1), F.softmax(q_output,dim=-1))+KLDivLoss(F.log_softmax(q_output,dim=-1), F.softmax(p_output,dim=-1))

    def forward(self, images, labels):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self.targeted:
            labels = self.target*torch.ones_like(labels)

        #loss = self.loss
        momentum = torch.zeros_like(images).detach().to(self.device)
        stacked_kernel = self.stacked_kernel.to(self.device)

        adv_delta = torch.zeros_like(images).to(self.device)
        
        clean_images = images.clone().detach()
        clean_images_shifted = clean_images+self.sigma*(torch.randn_like(clean_images))
        clean_images_shifted = torch.clamp(clean_images_shifted, min=0, max=1)
        if self.random_start:
            # Starting at a uniformly random point
            adv_delta = adv_delta + torch.empty_like(adv_delta).uniform_(-self.eps, self.eps)
            clean_images = torch.clamp(clean_images+adv_delta, min=0, max=1).detach()

        for _ in range(self.steps):
            #adv_images.requires_grad = True
            
            
            if _ >= self.k_ls:
                
                n_rap = torch.zeros_like(images).to(self.device)
                
                momentum_rap = torch.zeros_like(images).detach().to(self.device)
                for i in range(1):
                    n_rap.requires_grad = True
                    adv_images = torch.clamp(clean_images+adv_delta+n_rap, min=0, max=1)
                    if self.normalize:
                        outputs = self.model(self.norm(self.input_diversity(adv_images)))
                    else:
                        outputs = self.model(self.input_diversity(adv_images))
                    # Calculate loss
                    if self.targeted:
                        cost = self.logit_loss(outputs, labels)
                    else:
                        cost = -self.logit_loss(outputs, labels)
                    grad = torch.autograd.grad(cost, n_rap,
                                            retain_graph=False, create_graph=False)[0]
                    # depth wise conv2d
                    grad = F.conv2d(grad, stacked_kernel, stride=1, padding='same', groups=3)
                    grad = grad / torch.mean(torch.abs(grad), dim=(1,2,3), keepdim=True)
                    grad = grad + momentum_rap*self.decay
                    momentum_rap = grad

                    n_rap = n_rap.detach() + self.alpha*grad.sign()
                    n_rap = torch.clamp(n_rap, min=-self.epsilon_n, max=self.epsilon_n)
            
            for j in range(1):
                adv_delta.requires_grad = True
                if _ >= self.k_ls:
                    adv_images = torch.clamp(clean_images+adv_delta+n_rap, min=0, max=1)
                    adv_images_shifted = torch.clamp(clean_images_shifted+adv_delta+n_rap, min=0, max=1)
                else:
                    adv_images = torch.clamp(clean_images+adv_delta, min=0, max=1)
                    adv_images_shifted = torch.clamp(clean_images_shifted+adv_delta, min=0, max=1)

                if self.normalize:
                    outputs = self.model(self.norm(self.input_diversity(adv_images)))
                    outputs_shifted = self.model(self.norm(self.input_diversity(adv_images_shifted)))
                else:
                    outputs = self.model(self.input_diversity(adv_images))
                    outputs_shifted = self.model(self.input_diversity(adv_images_shifted))
                # Calculate loss
                if self.targeted:
                    cost = -self.logit_loss(outputs, labels)-self.gamma*self.logit_loss(outputs_shifted, labels)-self.lmd*self.double_kl_div(outputs, outputs_shifted,gamma=self.gamma)
                else:
                    cost = self.logit_loss(outputs, labels)+self.gamma*self.logit_loss(outputs_shifted, labels)-self.lmd*self.double_kl_div(outputs, outputs_shifted,gamma=self.gamma)

                # Update adversarial images
                grad = torch.autograd.grad(cost, adv_delta,
                                        retain_graph=False, create_graph=False)[0]
                # depth wise conv2d
                grad = F.conv2d(grad, stacked_kernel, stride=1, padding='same', groups=3)
                grad = grad / torch.mean(torch.abs(grad), dim=(1,2,3), keepdim=True)
                grad = grad + momentum*self.decay
                momentum = grad

                adv_delta = adv_delta.detach() + self.alpha*grad.sign()
                adv_delta = torch.clamp(adv_delta, min=-self.eps, max=self.eps)
        adv_images = torch.clamp(images + adv_delta, min=0, max=1).detach()

        return adv_images

    def kernel_generation(self):
        if self.kernel_name == 'gaussian':
            kernel = self.gkern(self.len_kernel, self.nsig).astype(np.float32)
        elif self.kernel_name == 'linear':
            kernel = self.lkern(self.len_kernel).astype(np.float32)
        elif self.kernel_name == 'uniform':
            kernel = self.ukern(self.len_kernel).astype(np.float32)
        else:
            raise NotImplementedError

        stack_kernel = np.stack([kernel, kernel, kernel])
        stack_kernel = np.expand_dims(stack_kernel, 1)
        return stack_kernel

    def gkern(self, kernlen=15, nsig=3):
        """Returns a 2D Gaussian kernel array."""
        x = np.linspace(-nsig, nsig, kernlen)
        kern1d = st.norm.pdf(x)
        kernel_raw = np.outer(kern1d, kern1d)
        kernel = kernel_raw / kernel_raw.sum()
        return kernel

    def ukern(self, kernlen=15):
        kernel = np.ones((kernlen,kernlen))* 1.0 /(kernlen*kernlen)
        return kernel

    def lkern(self, kernlen=15):
        kern1d = 1-np.abs(np.linspace((-kernlen+1)/2, (kernlen-1)/2, kernlen)/(kernlen+1)*2)
        kernel_raw = np.outer(kern1d, kern1d)
        kernel = kernel_raw / kernel_raw.sum()
        return kernel

    def input_diversity(self, x):
        img_size = x.shape[-1]
        img_resize = int(img_size * self.resize_rate)

        if self.resize_rate < 1:
            img_size = img_resize
            img_resize = x.shape[-1]

        rnd = torch.randint(low=img_size, high=img_resize, size=(1,), dtype=torch.int32)
        rescaled = F.interpolate(x, size=[rnd, rnd], mode='bilinear', align_corners=False)
        h_rem = img_resize - rnd
        w_rem = img_resize - rnd
        pad_top = torch.randint(low=0, high=h_rem.item(), size=(1,), dtype=torch.int32)
        pad_bottom = h_rem - pad_top
        pad_left = torch.randint(low=0, high=w_rem.item(), size=(1,), dtype=torch.int32)
        pad_right = w_rem - pad_left

        padded = F.pad(rescaled, [pad_left.item(), pad_right.item(), pad_top.item(), pad_bottom.item()], value=0)

        return padded if torch.rand(1) < self.diversity_prob else x


class IBTA_S2I_MIFGSM(Attack):

    def __init__(self, model, eps=8/255, alpha=2/255, steps=5, decay=1.0,loss=nn.CrossEntropyLoss(),targeted=False, target=None, normalize = True, rho = 0.5, N = 20, sigma = 16/255, diversity_prob = 0.7, resize_rate = 0.9, gamma = 0.1, lmd=0.1, sig=0.1):
        super().__init__("IBTA_S2I_MIFGSM", model)
        self.eps = eps
        self.steps = steps
        self.decay = decay
        self.alpha = alpha
        self.loss = loss
        self.targeted = targeted
        self.target = target
        self.normalize = normalize
        self.rho = rho
        self.N = N
        self.sigma = sigma
        self.diversity_prob = diversity_prob
        self.resize_rate = resize_rate
        self.gamma = gamma
        self.lmd = lmd
        self.sig = sig


    def norm(self, t):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        t[:, 0, :, :] = (t[:, 0, :, :] - mean[0])/std[0]
        t[:, 1, :, :] = (t[:, 1, :, :] - mean[1])/std[1]
        t[:, 2, :, :] = (t[:, 2, :, :] - mean[2])/std[2]

        return t

    def double_kl_div(self, p_output, q_output, get_softmax=True,gamma=0.01):

        KLDivLoss = nn.KLDivLoss(reduction='batchmean')
        
        return  gamma*KLDivLoss(F.log_softmax(p_output,dim=-1), F.softmax(q_output,dim=-1))+KLDivLoss(F.log_softmax(q_output,dim=-1), F.softmax(p_output,dim=-1))

    def input_diversity(self, x):
            img_size = x.shape[-1]
            img_resize = int(img_size * self.resize_rate)

            if self.resize_rate < 1:
                img_size = img_resize
                img_resize = x.shape[-1]

            rnd = torch.randint(low=img_size, high=img_resize, size=(1,), dtype=torch.int32)
            rescaled = F.interpolate(x, size=[rnd, rnd], mode='bilinear', align_corners=False)
            h_rem = img_resize - rnd
            w_rem = img_resize - rnd
            pad_top = torch.randint(low=0, high=h_rem.item(), size=(1,), dtype=torch.int32)
            pad_bottom = h_rem - pad_top
            pad_left = torch.randint(low=0, high=w_rem.item(), size=(1,), dtype=torch.int32)
            pad_right = w_rem - pad_left

            padded = F.pad(rescaled, [pad_left.item(), pad_right.item(), pad_top.item(), pad_bottom.item()], value=0)

            return padded if torch.rand(1) < self.diversity_prob else x    

    def forward(self, images, labels):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self.targeted:
            labels = self.target*torch.ones_like(labels)

        momentum = torch.zeros_like(images).detach().to(self.device)

        loss = self.loss

        adv_delta = torch.zeros_like(images).to(self.device)
        
        clean_images = images.clone().detach()
        clean_images_shifted = clean_images+self.sig*(torch.randn_like(clean_images))
        clean_images_shifted = torch.clamp(clean_images_shifted, min=0, max=1)

        for _ in range(self.steps):
            adv_delta.requires_grad = True
            adv_images = torch.clamp(clean_images+adv_delta, min=0, max=1)
            adv_images_shifted = torch.clamp(clean_images_shifted+adv_delta, min=0, max=1)
            noise = 0
            for n in range(self.N):
                gauss = torch.randn_like(clean_images) * (self.sigma / 255)
                
                x_dct = dct_2d(clean_images + gauss)
                x_dct_shifted = dct_2d(clean_images_shifted + gauss)
                mask = (torch.rand_like(clean_images) * 2 * self.rho + 1 - self.rho).cuda()
                x_idct = idct_2d(x_dct * mask) + adv_delta
                x_idct_shifted = idct_2d(x_dct_shifted * mask) + adv_delta

                # DI-FGSM https://arxiv.org/abs/1803.06978
                # output_v3 = model(DI(x_idct))
                if self.normalize:
                    outputs = self.model(self.norm(self.input_diversity(x_idct)))
                    outputs_shifted = self.model(self.norm(self.input_diversity(x_idct_shifted)))
                else:
                    outputs = self.model(self.input_diversity(x_idct))
                    outputs_shifted = self.model(self.input_diversity(x_idct_shifted))

                # Calculate loss
                if self.targeted:
                    cost = -loss(outputs, labels)-self.gamma*loss(outputs_shifted, labels)-self.lmd*self.double_kl_div(outputs, outputs_shifted,gamma=self.gamma)
                else:
                    cost = loss(outputs, labels)+self.gamma*loss(outputs_shifted, labels)-self.lmd*self.double_kl_div(outputs, outputs_shifted,gamma=self.gamma)

                # Update adversarial images
                grad_x = torch.autograd.grad(cost, adv_delta,
                                        retain_graph=False, create_graph=False)[0]
                grad_x = grad_x / torch.mean(torch.abs(grad_x), dim=(1,2,3), keepdim=True)
                
                noise += grad_x

            grad = noise / self.N
            grad = grad + momentum*self.decay
            momentum = grad

            adv_delta = adv_delta.detach() + self.alpha*grad.sign()
            adv_delta = torch.clamp(adv_delta, min=-self.eps, max=self.eps)
        
            #adv_images.requires_grad = True
            

        adv_images = torch.clamp(images + adv_delta, min=0, max=1).detach()

        return adv_images

class IBTA_DIFGSM(Attack):

    def __init__(self, model, eps=8/255, alpha=2/255, steps=10, decay=1.0,
                 resize_rate=0.9, diversity_prob=0.7, random_start=False,loss=nn.CrossEntropyLoss(),targeted=False, target=None, normalize=True, gamma=0.1, lmd=0.1, sigma=0.1):
        super().__init__("IBTA_DIFGSM", model)
        self.eps = eps
        self.steps = steps
        self.decay = decay
        self.alpha = alpha
        self.resize_rate = resize_rate
        self.diversity_prob = diversity_prob
        self.random_start = random_start
        self.supported_mode = ['default', 'targeted']
        self.loss = loss
        self.targeted = targeted
        self.target = target
        self.normalize = normalize
        self.gamma = gamma
        self.lmd = lmd
        self.sigma = sigma
    def norm(self, t):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        t[:, 0, :, :] = (t[:, 0, :, :] - mean[0])/std[0]
        t[:, 1, :, :] = (t[:, 1, :, :] - mean[1])/std[1]
        t[:, 2, :, :] = (t[:, 2, :, :] - mean[2])/std[2]

        return t
    def input_diversity(self, x):
        img_size = x.shape[-1]
        img_resize = int(img_size * self.resize_rate)

        if self.resize_rate < 1:
            img_size = img_resize
            img_resize = x.shape[-1]

        rnd = torch.randint(low=img_size, high=img_resize, size=(1,), dtype=torch.int32)
        rescaled = F.interpolate(x, size=[rnd, rnd], mode='bilinear', align_corners=False)
        h_rem = img_resize - rnd
        w_rem = img_resize - rnd
        pad_top = torch.randint(low=0, high=h_rem.item(), size=(1,), dtype=torch.int32)
        pad_bottom = h_rem - pad_top
        pad_left = torch.randint(low=0, high=w_rem.item(), size=(1,), dtype=torch.int32)
        pad_right = w_rem - pad_left

        padded = F.pad(rescaled, [pad_left.item(), pad_right.item(), pad_top.item(), pad_bottom.item()], value=0)

        return padded if torch.rand(1) < self.diversity_prob else x

    def double_kl_div(self, p_output, q_output, get_softmax=True, gamma=0.01):

        KLDivLoss = nn.KLDivLoss(reduction='batchmean')
        
        return gamma*KLDivLoss(F.log_softmax(p_output,dim=-1), F.softmax(q_output,dim=-1))+KLDivLoss(F.log_softmax(q_output,dim=-1), F.softmax(p_output,dim=-1))

    def forward(self, images, labels):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self.targeted:
            labels = self.target*torch.ones_like(labels)

        momentum = torch.zeros_like(images).detach().to(self.device)
        loss = self.loss
        adv_delta = torch.zeros_like(images).to(self.device)
        
        clean_images = images.clone().detach()
        clean_images_shifted = clean_images+self.sigma*(torch.randn_like(clean_images))
        clean_images_shifted = torch.clamp(clean_images_shifted, min=0, max=1).detach()
        if self.random_start:
            # Starting at a uniformly random point
            adv_delta = adv_delta + torch.empty_like(adv_delta).uniform_(-self.eps, self.eps)
            clean_images = torch.clamp(clean_images+adv_delta, min=0, max=1).detach()

        for _ in range(self.steps):
            adv_delta.requires_grad = True
            adv_images = torch.clamp(clean_images+adv_delta, min=0, max=1)
            adv_images_shifted = torch.clamp(clean_images_shifted+adv_delta, min=0, max=1)
            
            if self.normalize:
                outputs = self.model(self.norm(self.input_diversity(adv_images)))
                outputs_shifted = self.model(self.norm(self.input_diversity(adv_images_shifted)))
            else:
                outputs = self.model(self.input_diversity(adv_images))
                outputs_shifted = self.model(adv_images_shifted)
            # Calculate loss
            if self.targeted:
                cost = -loss(outputs, labels)-self.gamma*loss(outputs_shifted, labels)-self.lmd*self.double_kl_div(outputs, outputs_shifted, gamma=self.gamma)
            else:
                cost = loss(outputs, labels)+self.gamma*loss(outputs_shifted, labels)-self.lmd*self.double_kl_div(outputs, outputs_shifted, gamma=self.gamma)

            # Update adversarial images
            grad = torch.autograd.grad(cost, adv_delta,
                                       retain_graph=False, create_graph=False)[0]

            grad = grad / torch.mean(torch.abs(grad), dim=(1,2,3), keepdim=True)
            grad = grad + momentum*self.decay
            momentum = grad

            adv_delta = adv_delta.detach() + self.alpha*grad.sign()
            adv_delta = torch.clamp(adv_delta, min=-self.eps, max=self.eps)
        adv_images = torch.clamp(images + adv_delta, min=0, max=1).detach()

        return adv_images

class IBTA_MIFGSM(Attack):

    def __init__(self, model, eps=8/255, alpha=2/255, steps=5, decay=1.0,loss=nn.CrossEntropyLoss(),targeted=False, target=None, normalize = True, gamma=0.1, lmd=0.1, sigma=0.1):
        super().__init__("IBTA_MIFGSM", model)
        self.eps = eps
        self.steps = steps
        self.decay = decay
        self.alpha = alpha
        self.loss = loss
        self.targeted = targeted
        self.target = target
        self.normalize = normalize
        self.gamma = gamma
        self.lmd = lmd
        self.sigma = sigma
    def norm(self, t):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        t[:, 0, :, :] = (t[:, 0, :, :] - mean[0])/std[0]
        t[:, 1, :, :] = (t[:, 1, :, :] - mean[1])/std[1]
        t[:, 2, :, :] = (t[:, 2, :, :] - mean[2])/std[2]

        return t
    def double_kl_div(self, p_output, q_output, get_softmax=True,gamma=0.01):

        KLDivLoss = nn.KLDivLoss(reduction='batchmean')
        
        return  gamma*KLDivLoss(F.log_softmax(p_output,dim=-1), F.softmax(q_output,dim=-1))+KLDivLoss(F.log_softmax(q_output,dim=-1), F.softmax(p_output,dim=-1))
    def forward(self, images, labels):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self.targeted:
            labels = self.target*torch.ones_like(labels)

        momentum = torch.zeros_like(images).detach().to(self.device)

        loss = self.loss

        adv_delta = torch.zeros_like(images).to(self.device)
        
        clean_images = images.clone().detach()
        clean_images_shifted = clean_images+self.sigma*(torch.randn_like(clean_images))
        clean_images_shifted = torch.clamp(clean_images_shifted, min=0, max=1)
        for _ in range(self.steps):
            #adv_images.requires_grad = True
            adv_delta.requires_grad = True
            adv_images = torch.clamp(clean_images+adv_delta, min=0, max=1)
            adv_images_shifted = torch.clamp(clean_images_shifted+adv_delta, min=0, max=1)
            if self.normalize:
                outputs = self.model(self.norm(adv_images))
                outputs_shifted = self.model(self.norm(adv_images_shifted))
            else:
                outputs = self.model(adv_images)
                outputs_shifted = self.model(adv_images_shifted)
            # Calculate loss
            if self.targeted:
                cost = -loss(outputs, labels)-self.gamma*loss(outputs_shifted, labels)-self.lmd*self.double_kl_div(outputs, outputs_shifted,gamma=self.gamma)
            else:
                cost = loss(outputs, labels)+self.gamma*loss(outputs_shifted, labels)-self.lmd*self.double_kl_div(outputs, outputs_shifted,gamma=self.gamma)

            # Update adversarial images
            grad = torch.autograd.grad(cost, adv_delta,
                                       retain_graph=False, create_graph=False)[0]

            grad = grad / torch.mean(torch.abs(grad), dim=(1,2,3), keepdim=True)
            grad = grad + momentum*self.decay
            momentum = grad

            adv_delta = adv_delta.detach() + self.alpha*grad.sign()
            adv_delta = torch.clamp(adv_delta, min=-self.eps, max=self.eps)
        adv_images = torch.clamp(images + adv_delta, min=0, max=1).detach()

        return adv_images


class IBTA_TIFGSM(Attack):
    
    def __init__(self, model, eps=8/255, alpha=2/255, steps=20, decay=1.0, kernel_name='gaussian',
                 len_kernel=5, nsig=3, resize_rate=0.9, diversity_prob=0.7, p=0.5, random_start=False,
                 loss=nn.CrossEntropyLoss(),targeted=False, target=None,normalize=True, gamma=0.1, lmd=0.1, sigma=0.1):
        super().__init__("IBTA_TIFGSM", model)
        self.eps = eps
        self.steps = steps
        self.decay = decay
        self.alpha = alpha
        self.resize_rate = resize_rate
        self.diversity_prob = diversity_prob
        self.random_start = random_start
        self.kernel_name = kernel_name
        self.len_kernel = len_kernel
        self.nsig = nsig
        self.stacked_kernel = torch.from_numpy(self.kernel_generation())
        self.p = p
        self.loss = loss
        self.targeted = targeted
        self.target = target
        self.normalize = normalize
        self.lmd = lmd
        self.gamma = gamma
        self.sigma = sigma
    def norm(self, t):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        t[:, 0, :, :] = (t[:, 0, :, :] - mean[0])/std[0]
        t[:, 1, :, :] = (t[:, 1, :, :] - mean[1])/std[1]
        t[:, 2, :, :] = (t[:, 2, :, :] - mean[2])/std[2]

        return t
    def double_kl_div(self, p_output, q_output, get_softmax=True, gamma=0.01):

        KLDivLoss = nn.KLDivLoss(reduction='batchmean')
        
        return gamma*KLDivLoss(F.log_softmax(p_output,dim=-1), F.softmax(q_output,dim=-1))+KLDivLoss(F.log_softmax(q_output,dim=-1), F.softmax(p_output,dim=-1))
    def random_region_mask(self, input_tensor, region_size=128, mask_ratio=0.3):
        output_tensor = input_tensor.clone()  
        b, c, h, w = input_tensor.shape
        
        x = random.randint(0, h - region_size)
        y = random.randint(0, w - region_size)

        num_elements = int(region_size * region_size * mask_ratio)

        indices = random.sample(range(region_size * region_size), num_elements)

        for idx in indices:
            i = idx // region_size
            j = idx % region_size
            output_tensor[:, :, x+i, y+j] = 0

        return output_tensor
    def forward(self, images, labels):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self.targeted:
            labels = self.target*torch.ones_like(labels)

        loss = self.loss
        momentum = torch.zeros_like(images).detach().to(self.device)
        stacked_kernel = self.stacked_kernel.to(self.device)

        adv_delta = torch.zeros_like(images).to(self.device)
        
        clean_images = images.clone().detach()
        
        clean_images_masked = clean_images+self.sigma*(torch.randn_like(clean_images))
        clean_images_masked = torch.clamp(clean_images_masked, min=0, max=1)
        if self.random_start:
            # Starting at a uniformly random point
            adv_delta = adv_delta + torch.empty_like(adv_delta).uniform_(-self.eps, self.eps)
            clean_images = torch.clamp(clean_images+adv_delta, min=0, max=1).detach()

        for _ in range(self.steps):
            #adv_images.requires_grad = True
            
            adv_delta.requires_grad = True
            adv_images = torch.clamp(clean_images+adv_delta, min=0, max=1)
            adv_images_masked = torch.clamp(clean_images_masked+adv_delta, min=0, max=1)
            if self.normalize:
                outputs = self.model(self.norm(self.input_diversity(adv_images)))
                outputs_masked = self.model(self.norm(self.input_diversity(adv_images_masked)))
            else:
                outputs = self.model(self.input_diversity(adv_images))
                outputs_masked = self.model(self.input_diversity(adv_images_masked))

            # Calculate loss
            if self.targeted:
                cost = -loss(outputs, labels)-self.gamma*loss(outputs_masked, labels)-self.lmd*self.double_kl_div(outputs, outputs_masked,gamma=self.gamma)
                
            else:
                cost = loss(outputs, labels)+self.gamma*loss(outputs_masked, labels)-self.lmd*self.double_kl_div(outputs, outputs_masked,gamma=self.gamma)
                

            # Update adversarial images
            grad = torch.autograd.grad(cost, adv_delta,
                                       retain_graph=False, create_graph=False)[0]
            # depth wise conv2d
            grad = F.conv2d(grad, stacked_kernel, stride=1, padding='same', groups=3)
            grad = grad / torch.mean(torch.abs(grad), dim=(1,2,3), keepdim=True)
            grad = grad + momentum*self.decay
            momentum = grad

            adv_delta = adv_delta.detach() + self.alpha*grad.sign()
            adv_delta = torch.clamp(adv_delta, min=-self.eps, max=self.eps)
        adv_images = torch.clamp(images + adv_delta, min=0, max=1).detach()

        return adv_images

    def kernel_generation(self):
        if self.kernel_name == 'gaussian':
            kernel = self.gkern(self.len_kernel, self.nsig).astype(np.float32)
        elif self.kernel_name == 'linear':
            kernel = self.lkern(self.len_kernel).astype(np.float32)
        elif self.kernel_name == 'uniform':
            kernel = self.ukern(self.len_kernel).astype(np.float32)
        else:
            raise NotImplementedError

        stack_kernel = np.stack([kernel, kernel, kernel])
        stack_kernel = np.expand_dims(stack_kernel, 1)
        return stack_kernel

    def gkern(self, kernlen=15, nsig=3):
        """Returns a 2D Gaussian kernel array."""
        x = np.linspace(-nsig, nsig, kernlen)
        kern1d = st.norm.pdf(x)
        kernel_raw = np.outer(kern1d, kern1d)
        kernel = kernel_raw / kernel_raw.sum()
        return kernel

    def ukern(self, kernlen=15):
        kernel = np.ones((kernlen,kernlen))* 1.0 /(kernlen*kernlen)
        return kernel

    def lkern(self, kernlen=15):
        kern1d = 1-np.abs(np.linspace((-kernlen+1)/2, (kernlen-1)/2, kernlen)/(kernlen+1)*2)
        kernel_raw = np.outer(kern1d, kern1d)
        kernel = kernel_raw / kernel_raw.sum()
        return kernel

    def input_diversity(self, x):
        img_size = x.shape[-1]
        img_resize = int(img_size * self.resize_rate)

        if self.resize_rate < 1:
            img_size = img_resize
            img_resize = x.shape[-1]

        rnd = torch.randint(low=img_size, high=img_resize, size=(1,), dtype=torch.int32)
        rescaled = F.interpolate(x, size=[rnd, rnd], mode='bilinear', align_corners=False)
        h_rem = img_resize - rnd
        w_rem = img_resize - rnd
        pad_top = torch.randint(low=0, high=h_rem.item(), size=(1,), dtype=torch.int32)
        pad_bottom = h_rem - pad_top
        pad_left = torch.randint(low=0, high=w_rem.item(), size=(1,), dtype=torch.int32)
        pad_right = w_rem - pad_left

        padded = F.pad(rescaled, [pad_left.item(), pad_right.item(), pad_top.item(), pad_bottom.item()], value=0)

        return padded if torch.rand(1) < self.diversity_prob else x

