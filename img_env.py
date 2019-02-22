import numpy as np

import torch
from gym.spaces import Discrete, Box
import torchvision.transforms as T
from torchvision import datasets

from cityscapes import Cityscapes
from torchvision import transforms
from transforms import RandomNoise

import matplotlib.pyplot as plt

CITYSCAPE = '/datasets01/cityscapes/112817/gtFine'
IMG_ENVS = ['mnist', 'cifar10', 'cifar100', 'imagenet']


def get_data_loader(env_id, train=True):
    kwargs = {'num_workers': 0, 'pin_memory': True}
    # transform = T.Compose(
    #     [T.ToTensor()])
    transform = T.Compose(
        [T.ToTensor(), RandomNoise(),
         T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    original_transform = T.Compose([T.ToTensor()])
    #transform = T.Compose([T.ToTensor()])
    # print(env_id)
    if env_id in IMG_ENVS:
        if env_id == 'mnist':
            # print("entered")
            transform = T.Compose([
                           T.Resize(size=(32, 32)),
                           T.ToTensor(),
                           T.Normalize((0.1307,), (0.3081,))
                       ])
            dataset = datasets.MNIST
        elif env_id == 'cifar10':
            dataset = datasets.CIFAR10
        elif env_id == 'cifar100':
            dataset = datasets.CIFAR100
        elif env_id == 'imagenet':
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

            if train:
                data_dir = ''
            else:
                data_dir = ''
            dataset = datasets.ImageFolder(
                data_dir,
                transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]))
        original_loader = torch.utils.data.DataLoader(
            dataset('data', train=train, download=True, transform = original_transform))
        loader = torch.utils.data.DataLoader(
            dataset('data', train=train, download=True,
                    transform=transform),
            batch_size=1, shuffle=True, **kwargs)
    elif env_id in ['cityscapes']:
        loader = torch.utils.data.DataLoader(
            Cityscapes(CITYSCAPE, train, transform=transform),
            batch_size=1, shuffle=True, **kwargs)
    return loader, original_loader


class ImgEnv(object):
    def __init__(self, dataset, train, max_steps, channels, window=5, random_label=0.):
        self.action_space = Discrete(4)
        self.observation_space = Box(low=0, high=1, shape=(channels, 32, 32))
        self.channels = channels
        self.data_loader, self.original_loader = get_data_loader(dataset, train=train)
        self.window = window
        self.max_steps = max_steps
        self.labels_list = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

    def seed(self, seed):
        np.random.seed(seed)

    def display_original(self, num_update):
        org_curr_img, org_curr_label = next(iter(self.original_loader))

        img = transforms.ToPILImage()(self.curr_img)
        #print("ORIGINAL IMG", org_curr_img)
        #print("TYPE:", org_curr_img.shape)
        img.save("state_cifar/"+"image_test_label_update_" + str(num_update) + "_"+str(self.labels_list[int(self.curr_label.item())])+"_original"+".png")



    def display_step(self, step_number, num_update):
        display_state = self.curr_img
        for i in range(self.curr_img.shape[0]):
            for j in range(self.curr_img.shape[1]):
                for k in range(self.curr_img.shape[2]):
                    if self.state[0][j][k] == 0:
                        display_state[i][j][k] = 0

        img = transforms.ToPILImage()(display_state)
        img.save("state_cifar/"+"image_test_label_update_" + str(num_update) + "_"+str(self.labels_list[int(self.curr_label.item())])+"_step_"+str(step_number)+".png")


    def reset(self):
        self.curr_img, self.curr_label = next(iter(self.data_loader))
        

        self.curr_img = self.curr_img.squeeze(0)
        self.curr_label = self.curr_label.squeeze(0)
        img = transforms.ToPILImage()(self.curr_img)
        print("IMG:", self.curr_img.shape)
        img.save("state_cifar/"+"original_" +".png")
        # plt.imshow(self.curr_img.numpy().transpose((1, 2, 0)))
        # plt.savefig("og.png")
        # plt.close()

        #print("IMAGE", self.curr_img.shape)
        
        #print("LABEL", self.curr_label.item())
        
        # initialize position at center of image
        self.pos = [max(0, self.curr_img.shape[1]//2-self.window), max(0, self.curr_img.shape[2]//2-self.window)]
        self.state = -np.ones(
            (self.channels, self.curr_img.shape[1], self.curr_img.shape[2]))
        self.state[0, :, :] = np.zeros(
            (1, self.curr_img.shape[1], self.curr_img.shape[2]))
        self.state[0, self.pos[0]:self.pos[0]+self.window, self.pos[1]:self.pos[1]+self.window] = 1
        self.state[
            1:, self.pos[0]:self.pos[0]+self.window, self.pos[1]:self.pos[1]+self.window] = \
            self.curr_img[:, self.pos[0]:self.pos[0]+self.window, self.pos[1]:self.pos[1]+self.window]
        self.num_steps = 0

        # print("LABEL", self.curr_label.item())
        return self.state

    def step(self, action):
        done = False
        # Go left
        action = action[0]
        #print(action)
        if action[0] == 0:
            self.pos[0] = max(0, self.pos[0])
        # Go right
        elif action[0] == 1:
            self.pos[0] = min(self.curr_img.shape[1] - 1,
                              self.pos[0] + self.window)
        # Go up
        elif action[0] == 2:
            self.pos[1] = max(0, self.pos[1])

        # Go down
        elif action[0] == 3:
            self.pos[1] = min(self.curr_img.shape[2] - 1,
                              self.pos[1] + self.window)

        # elif action == 4:
        #     done = True
        else:
            print("Action out of bounds!")
            return
        self.state[0, :, :] = np.zeros(
            (1, self.curr_img.shape[1], self.curr_img.shape[2]))
        self.state[0, self.pos[0]:self.pos[0]+self.window, self.pos[1]:self.pos[1]+self.window] = 1
        self.state[1:,
            self.pos[0]:self.pos[0]+self.window, self.pos[1]:self.pos[1]+self.window] = \
                self.curr_img[:, self.pos[0]:self.pos[0]+self.window, self.pos[1]:self.pos[1]+self.window]
        self.num_steps += 1
        done = self.num_steps >= self.max_steps
        reward = -0.1
        if action[1] == self.curr_label.item():
            reward = 1
        if reward > 0:
            done = True
        return self.state, reward, done, {}

    def get_current_obs(self):
        return self.state

    def close(self):
        pass


class DetectionEnv(object):
    def __init__(self, dataset, train, max_steps):
        self.action_space = Discrete(4)
        self.channels = 4
        self.observation_space = Box(
            low=0, high=1, shape=(self.channels, 256, 256))
        self.max_steps = max_steps
        self.seed(0)
        self.loader = get_data_loader(dataset, train=train)
        self.window = 10

    def seed(self, seed):
        np.random.seed(seed)

    def reset(self):
        # initialize position at center of image
        self.curr_img, self.curr_mask = iter(self.loader).next()
        self.curr_img = self.curr_img.squeeze(0)
        self.curr_mask = self.curr_mask.squeeze(0)
        unique_objects = np.unique(self.curr_mask)

        self.pos = [self.curr_img.shape[1]//2, self.curr_img.shape[2]//2]
        self.state = np.zeros(
            (self.channels, self.curr_img.shape[1], self.curr_img.shape[2]))
        self.state[0, self.pos[0]:self.pos[0]+self.window, self.pos[1]:self.pos[1]+self.window] = 1
        self.state[1:, :, :] = self.curr_img
        curr_obj = self.curr_mask[self.pos]
        unique_objects = np.delete(unique_objects, curr_obj)
        if len(unique_objects) > 0:
            self.goal = np.random.choice(unique_objects)
        else:
            self.reset()
        self.num_steps = 0
        return self.state

    def step(self, action):
        # Go left
        if action == 0:
            self.pos[0] = max(0, self.pos[0])
        # Go right
        elif action == 1:
            self.pos[0] = min(self.curr_img.shape[1] - 1,
                              self.pos[0] + self.window)
        # Go up
        elif action == 2:
            self.pos[1] = max(0, self.pos[1])

        # Go down
        elif action == 3:
            self.pos[1] = min(self.curr_img.shape[2] - 1,
                              self.pos[1] + self.window)

        else:
            print("Action out of bounds!")
            return
        self.state[0, :, :] = np.zeros(
            (1, self.curr_img.shape[1], self.curr_img.shape[2]))
        self.state[
            0, self.pos[0]:self.pos[0]+self.window,
            self.pos[1]:self.pos[1]+self.window] = 1
        self.num_steps += 1
        done = self.num_steps >= self.max_steps
        reward = 0
        if self.check_done():
            done = True
            reward = 1
            

        return self.state, reward, done, {}

    def check_done(self):
        objects = np.unique(
            self.curr_mask[
                self.pos[0] - self.window:self.pos[0] + self.window,
                self.pos[1] - self.window:self.pos[1] + self.window])
        if self.goal in objects:
            return True
        return False

    def get_current_obs(self):
        return self.state

    def close(self):
        pass


if __name__ == '__main__':
    transform = T.Compose([
                   T.ToTensor(),
                   T.Normalize((0.1307,), (0.3081,))
               ])
    dataset = datasets.MNIST
    channels = 2
    train = True
    loader = torch.utils.data.DataLoader(
        dataset('data', train=train, download=True,
            transform=transform),
        batch_size=60000, shuffle=True)
    for imgs, labels in loader:
        break
    #print(labels)
    #env = ImgEnv(imgs, labels, max_steps=200, channels=channels, window=5)
    env = ImgEnv('mnist', True, max_steps=200, channels=channels, window=5)

    #print(env)

    # loader = torch.utils.data.DataLoader(
    #     Cityscapes(CITYSCAPE, train, transform=transform), batch_size=10000,
    #     shuffle=True)
    # for imgs, labels in loader:
    #     break
    # #env = DetectionEnv(imgs, labels, max_steps=200)
    # env = DetectionEnv('mnist', True, max_steps=200)





