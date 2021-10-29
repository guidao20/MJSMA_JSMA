import numpy as np 
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F  
import torchvision  
from torchvision import transforms 
import matplotlib.pyplot as plt 
from torch.autograd.gradcheck import zero_gradients
import os

## Network structure
class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.fc1 = nn.Linear(784, 300)
		self.fc2 = nn.Linear(300, 100)
		self.fc3 = nn.Linear(100, 10)
	def forward(self, x):
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x

def compute_jacobian(model, input):
	output = model(input)
	num_features = int(np.prod(input.shape[1:]))
	jacobian = torch.zeros([output.size()[1], num_features])
	mask = torch.zeros(output.size())
	for i in range(output.size()[1]):
		mask[:,i] = 1
		zero_gradients(input)
		output.backward(mask, retain_graph=True)
		jacobian[i] = input._grad.squeeze().view(-1, num_features).clone()
		mask[:,i] = 0
	return jacobian


def max_perturbation_choose(Vector,gamma):
	mid_number = int(Vector.shape[1] * gamma)
	abs_vector = torch.abs(Vector)
	sort_vector = abs_vector.sort(1,True)[0]
	mid_value = (sort_vector[:,mid_number-2]+sort_vector[:,mid_number-1])/2
	one_matrix = torch.ones_like(Vector)
	shape_num = Vector.shape[0]
	compare_matrix = one_matrix*mid_value.view(shape_num,1)
	Mask_range = abs_vector.gt(compare_matrix)
	Vector_range = Vector * Mask_range
	return Vector_range

def saliency_map(jacobian, target_index, nb_features, targeted):
	all_sum = torch.sum(jacobian, dim=0, keepdim=True)
	target_grad = jacobian[target_index]
	others_grad = all_sum - target_grad
	adv_temp_map = others_grad * target_grad
	if targeted == "targeted":
		mask1 = target_grad.gt(0)        # find + perturbation
		mask2 = -1 * target_grad.lt(0)   # find - perturbation         
	else:
		mask1 = -1 * target_grad.gt(0)   # find - perturbation
		mask2 = target_grad.lt(0)        # find + perturbation
	mask3 = mask1 + mask2                # merge perturbation
	mask4 = adv_temp_map.lt(0)      # localtion perturbation
	adv_saliency_map = torch.abs(adv_temp_map) * mask4 * mask3
	return adv_saliency_map

def generate_adversarial_example(image, ys_target, gamma, model, targeted):
	copy_sample = np.copy(image)
	var_sample = Variable(torch.from_numpy(copy_sample), requires_grad=True)
	var_target = Variable(torch.LongTensor([ys_target,]))
	num_features = int(np.prod(copy_sample.shape[1:]))
	shape = var_sample.size()
	model.eval()
	output = model(var_sample)
	current = torch.max(output.data, 1)[1].numpy()
	jacobian = compute_jacobian(model, var_sample)
	adv_saliency_map = saliency_map(jacobian, var_target, num_features, targeted)
	adv_saliency_map = max_perturbation_choose(adv_saliency_map, 1.0)
	new_examples =  torch.clamp(adv_saliency_map * 5 + var_sample, 0.0 , 1.0)
	adversarial_examples = new_examples.view(shape)
	return adversarial_examples



if __name__ == "__main__":
	# Define data format
	mnist_transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x : x.resize_(28*28))])
	testdata = torchvision.datasets.MNIST(root="./mnist", train=False, download=True, transform=mnist_transform)
	testloader = torch.utils.data.DataLoader(testdata, batch_size=256, shuffle=True, num_workers=0)

	# Load model parameters
	net = torch.load('mnist_net_all.pkl')

	#choose sample
	index = 10 
	images = testdata[index][0].resize_(1,784).numpy()
	label = torch.tensor([testdata[index][1]])
	gamma = 1   # Perturbation rate
	ys_target = 0 # Adversarial label, if untarget, ys_target is the source label.
	targeted = 'untargeted'

	outputs = net(torch.from_numpy(np.copy(images)))
	predicted = torch.max(outputs.data, 1)[1]
	print('MJSMA无目标攻击前：\n 测试样本类别的预测值：{}'.format(predicted[0]))

	# Craft adversarial adversarial examples
	adversarial_examples = generate_adversarial_example(images, ys_target, gamma, net, targeted)

	outputs = net(adversarial_examples)
	predicted = torch.max(outputs.data, 1)[1]
	print('MJSMA无目标攻击后：\n 测试样本类别的预测值：{}'.format(predicted[0]))



