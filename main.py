


def custom_replace(tensor, on_zero, on_non_zero):
    # we create a copy of the original tensor, 
    # because of the way we are replacing them.
    res = tensor.clone()
    res[tensor==0] = on_zero
    res[tensor!=0] = on_non_zero
    return res

class Plot(object):
    def __init__(self, title, port=8080):
        self.viz = Visdom(port=port)
        self.windows = {}
        self.title = title

    def register_scatterplot(self, name, xlabel, ylabel):
        win = self.viz.scatter(
            X=numpy.zeros((1, 2)),
            opts=dict(title=self.title, markersize=5, xlabel=xlabel, ylabel=ylabel)
        )
        self.windows[name] = win

    def update_scatterplot(self, name, x, y):
        self.viz.updateTrace(
            X=numpy.array([x]),
            Y=numpy.array([y]),
            win=self.windows[name]
        )


def main():
    import copy
    import glob
    import os
    import time
    import matplotlib.pyplot as plt

    import gym
    import numpy as np
    import torch
    torch.multiprocessing.set_start_method('spawn')

    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from gym.spaces import Discrete

    from arguments import get_args
    from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
    from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
    from baselines.common.vec_env.vec_normalize import VecNormalize
    from envs import make_env
    from img_env import ImgEnv, IMG_ENVS
    from model import Policy
    from storage import RolloutStorage
    from utils import update_current_obs, eval_episode
    from torchvision import transforms
    from visdom import Visdom

    import algo

    viz = Visdom(port=8097)

    print("#######")
    print("WARNING: All rewards are clipped or normalized so you need to use a monitor (see envs.py) or visdom plot to get true rewards")
    print("#######")

    plot_rewards = []
    plot_policy_loss = []
    plot_value_loss = []
    # x = np.array([0])
    # y = np.array([0])
    # counter = 0
    # win = viz.line(
    #     X=x,
    #     Y=y,
    #     win="test1",
    #     name='Line1',
    #     opts=dict(
    #         title='Reward',
    #     )
    #     )
    # win2 = viz.line(
    #     X=x,
    #     Y=y,
    #     win="test2",
    #     name='Line2',
    #     opts=dict(
    #         title='Policy Loss',
    #     )
    #     )
    # win3 = viz.line(
    #     X=x,
    #     Y=y,
    #     win="test3",
    #     name='Line3',
    #     opts=dict(
    #         title='Value Loss',
    #     )
    #     )

    args = get_args()
    if args.no_cuda:
        args.cuda = False
    print(args)
    assert args.algo in ['a2c', 'ppo', 'acktr']
    if args.recurrent_policy:
        assert args.algo in ['a2c', 'ppo'], \
            'Recurrent policy is not implemented for ACKTR'

    num_updates = int(args.num_frames) // args.num_steps // args.num_processes

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    toprint = ['seed', 'lr', 'nat', 'resnet']
    if args.env_name in IMG_ENVS:
        toprint += ['window', 'max_steps']
    toprint.sort()
    name = args.tag
    args_param = vars(args)
    os.makedirs(os.path.join(args.out_dir, args.env_name), exist_ok=True)
    for arg in toprint:
        if arg in args_param and (args_param[arg] or arg in ['gamma', 'seed']):
            if args_param[arg] is True:
                name += '{}_'.format(arg)
            else:
                name += '{}{}_'.format(arg, args_param[arg])
    model_dir = os.path.join(args.out_dir, args.env_name, args.algo)
    os.makedirs(model_dir, exist_ok=True)

    results_dict = {
        'episodes': [],
        'rewards': [],
        'args': args
    }
    torch.set_num_threads(1)
    eval_env = make_env(args, 'cifar10', args.seed, 1, None,
            args.add_timestep, natural=args.nat, train=False)
    envs = make_env(args, 'cifar10', args.seed, 1, None,
            args.add_timestep, natural=args.nat, train=True)
                
    #print(envs)
    # envs = envs[0]
    

    # if args.num_processes > 1:
    #     envs = SubprocVecEnv(envs)
    # else:
    #     envs = DummyVecEnv(envs)
    # eval_env = DummyVecEnv(eval_env)
    # if len(envs.observation_space.shape) == 1:
    #     envs = VecNormalize(envs, gamma=args.gamma)

    obs_shape = envs.observation_space.shape
    obs_shape = (obs_shape[0] * args.num_stack, *obs_shape[1:])

    actor_critic = Policy(obs_shape, envs.action_space, args.recurrent_policy,
                          dataset=args.env_name, resnet=args.resnet,
                          pretrained=args.pretrained)
    if envs.action_space.__class__.__name__ == "Discrete":
        action_shape = 1
    else:
        action_shape = envs.action_space.shape[0]

    if args.cuda:
        actor_critic.cuda()

    if args.algo == 'a2c':
        agent = algo.A2C_ACKTR(actor_critic, args.value_loss_coef,
                               args.entropy_coef, lr=args.lr,
                               eps=args.eps, alpha=args.alpha,
                               max_grad_norm=args.max_grad_norm)
    elif args.algo == 'ppo':
        agent = algo.PPO(actor_critic, args.clip_param, args.ppo_epoch, args.num_mini_batch,
                         args.value_loss_coef, args.entropy_coef, lr=args.lr,
                               eps=args.eps,
                               max_grad_norm=args.max_grad_norm)
    elif args.algo == 'acktr':
        agent = algo.A2C_ACKTR(actor_critic, args.value_loss_coef,
                               args.entropy_coef, acktr=True)

    action_space = envs.action_space
    if args.env_name in IMG_ENVS:
        action_space = np.zeros(2)
    # obs_shape = envs.observation_space.shape
    rollouts = RolloutStorage(args.num_steps, args.num_processes, obs_shape, action_space, actor_critic.state_size)
    current_obs = torch.zeros(args.num_processes, *obs_shape)

    obs = envs.reset()

    update_current_obs(obs, current_obs, obs_shape, args.num_stack)
    rollouts.observations[0].copy_(current_obs)

    # These variables are used to compute average rewards for all processes.
    episode_rewards = torch.zeros([args.num_processes, 1])
    final_rewards = torch.zeros([args.num_processes, 1])

    if args.cuda:
        current_obs = current_obs.cuda()
        rollouts.cuda()

    start = time.time()
    for j in range(num_updates):
        #envs.display_original(j)
        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, states = actor_critic.act(
                        rollouts.observations[step],
                        rollouts.states[step],
                        rollouts.masks[step])
            cpu_actions = action.squeeze(1).cpu().numpy()

            # Obser reward and next obs
            obs, reward, done, info = envs.step(cpu_actions)

            # envs.display_step(step, j)

            # print("OBS", obs)

            # print("REWARD", reward)
            # print("DONE", done)
            # print("INFO", info)


            reward = torch.from_numpy(np.expand_dims(np.stack([reward]), 1)).float()
            episode_rewards += reward

            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in [done]])
            final_rewards *= masks
            final_rewards += (1 - masks) * episode_rewards
            episode_rewards *= masks

            if args.cuda:
                masks = masks.cuda()

            if current_obs.dim() == 4:
                current_obs *= masks.unsqueeze(2).unsqueeze(2)
            else:
                current_obs *= masks

            update_current_obs(obs, current_obs, obs_shape, args.num_stack)
            rollouts.insert(current_obs, states, action, action_log_prob, value, reward, masks)

            # print("envs.curr_img SHAPE: ", envs.curr_img.shape)
            #display_state = envs.curr_img
            # display_state[:, envs.pos[0]:envs.pos[0]+envs.window, envs.pos[1]:envs.pos[1]+envs.window] = 5
            # display_state = custom_replace(display_state, 1, 0)
            # display_state[:, envs.pos[0]:envs.pos[0]+envs.window, envs.pos[1]:envs.pos[1]+envs.window] = \
            #     envs.curr_img[:, envs.pos[0]:envs.pos[0]+envs.window, envs.pos[1]:envs.pos[1]+envs.window]
            # img = transforms.ToPILImage()(display_state)
            # img.save("state_cifar/"+"state"+str(j)+"_"+str(step)+".png")

        with torch.no_grad():
            next_value = actor_critic.get_value(rollouts.observations[-1],
                                                rollouts.states[-1],
                                                rollouts.masks[-1]).detach()

        rollouts.compute_returns(next_value, args.use_gae, args.gamma, args.tau)

        value_loss, action_loss, dist_entropy = agent.update(rollouts)

        rollouts.after_update()

        

        if j % args.save_interval == 0:
            torch.save((actor_critic.state_dict(), results_dict), os.path.join(
                model_dir, name + 'cifar_model2.pt'))

        if j % args.log_interval == 0:
            end = time.time()
            total_reward = eval_episode(eval_env, actor_critic, args)

            

            results_dict['rewards'].append(total_reward)
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            print("Updates {}, num timesteps {}, FPS {}, reward {:.1f} entropy {:.5f}, value loss {:.5f}, policy loss {:.5f}".
                format(j, total_num_steps,
                       int(total_num_steps / (end - start)),
                       np.mean(results_dict['rewards'][-10:]), dist_entropy,
                       value_loss, action_loss))

            # viz.scatter(
            #     X=(np.linspace(len(results_dict['rewards']),2)),
            #     Y=results_dict['rewards'],
            #     opts=dict(
            #         legend=['Reward'],
            #         xtickmin=0,
            #         xtickmax=len(results_dict['rewards']),
            #         xtickstep=1,
            #         ytickmin=min(results_dict['rewards']),
            #         ytickmax=max(results_dict['rewards']),
            #         ytickstep=0.05,
            #     ),
            # )
            plot_rewards.append(np.mean(results_dict['rewards'][-10:]))
            plt.plot(range(len(plot_rewards)), plot_rewards)
            plt.savefig("rewards.png")

            plot_policy_loss.append(action_loss)
            plt.plot(range(len(plot_policy_loss)), plot_policy_loss)
            plt.savefig("policyloss.png")

            plot_value_loss.append(value_loss)
            plt.plot(range(len(plot_value_loss)), plot_value_loss)
            plt.savefig("valueloss.png")
            # x = range(len(plot_rewards))
            # y = plot_rewards
            # viz.update(x, y)

            # viz.line(
            #     X=np.array([counter + 1]),
            #     Y=np.array([np.mean(results_dict['rewards'][-10:])]),
            #     win="test1",
            #     name='Line1',
            #     update='append',

            # )
            # viz.line(
            #     X=np.array([counter + 1]),
            #     Y=np.array([value_loss]),
            #     win="test2",
            #     name='Line2',
            #     update='append',

            # )
            # viz.line(
            #     X=np.array([counter + 1]),
            #     Y=np.array([action_loss]),
            #     win="test3",
            #     name='Line3',
            #     update='append',


            # )
            # counter = counter + 1


if __name__ == "__main__":
    main()
