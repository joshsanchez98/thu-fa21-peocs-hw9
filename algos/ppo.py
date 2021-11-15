from .base import *
from torch.distributions import Categorical

class PPO(Base):
	def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, 
		p_lr: float=7e-4, v_lr: float=1e-4, n_epochs: int=4, 
		eps_clip: float=0.2, entropy_coeff: float=0.01):
		
		super(PPO, self).__init__(
			input_dim=input_dim, 
			hidden_dim=hidden_dim, 
			output_dim=output_dim,
			p_lr=p_lr,
			v_lr=v_lr
		)

		# create online actor
		self.old_actor = models.FCNetwork(input_dim=input_dim, hidden_dim=hidden_dim, 
			output_dim=output_dim).to(self.device)
		self.old_actor.load_state_dict(self.actor.state_dict())
		
		# additional parameters to ppo
		self.n_epochs = n_epochs
		self.eps_clip = eps_clip
		self.entropy_coeff = entropy_coeff


	def update(self, buff):
		# get data from buffer
		s  = buff.get_states() 	
		a  = buff.get_actions()
		r  = buff.get_rewards()
		rg = buff.get_rewards2go()

		# pass state through actor and 
		# select the appropriate action
		with torch.no_grad():
			old_logits = self.old_actor(s)
			old_dist = F.softmax(old_logits, dim=1)
			old_prob = torch.gather(old_dist, 1, a).squeeze(1)

		for k in range(self.n_epochs):
			# pass state through actor and 
			# select the appropriate action
			logits = self.actor(s)
			dist = F.softmax(logits, dim=1)
			prob = torch.gather(dist, 1, a).squeeze(1)
			cat_dist = Categorical(logits=logits)

			# pass state throgh critic
			values = self.critic(s).squeeze(1)

			# update loss
			baseline = values.detach()
			adv = rg - baseline
			ratio1 = prob / old_prob.detach()
			ratio2 = torch.clamp(ratio1, 1 - self.eps_clip, 1 + self.eps_clip)
			actor_loss = -torch.min(ratio1 * adv, ratio2 * adv) - self.entropy_coeff * cat_dist.entropy()
			critic_loss = (values - rg)**2

			# optimization step
			self.optim_actor.zero_grad()
			self.optim_critic.zero_grad()
			actor_loss.mean().backward()
			critic_loss.mean().backward()
			self.optim_actor.step()
			self.optim_critic.step()

		# update old actor
		self.old_actor.load_state_dict(self.actor.state_dict())
