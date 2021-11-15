from .base import *
from torch.distributions import Categorical

class REINFORCE(Base):
	def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, 
		p_lr: float=7e-4, v_lr: float=1e-4):
		
		super(REINFORCE, self).__init__(
			input_dim=input_dim, 
			hidden_dim=hidden_dim, 
			output_dim=output_dim,
			p_lr=p_lr,
			v_lr=v_lr
		)		


	def update(self, buff):
		# get data from buffer
		s  = buff.get_states() 	
		a  = buff.get_actions()
		r  = buff.get_rewards()
		rg = buff.get_rewards2go()


		# pass state through actor and 
		# select the appropriate action
		logits = self.actor(s)
		log_dist = F.log_softmax(logits, dim=1)
		log_prob = torch.gather(log_dist, 1, a).squeeze(1)

		# pass state throgh critic
		values = self.critic(s).squeeze(1)

		# update loss
		baseline = values.detach()
		actor_loss = -log_prob * (rg - baseline) 
		critic_loss = (values - rg)**2

		# optimization step
		self.optim_actor.zero_grad()
		self.optim_critic.zero_grad()
		actor_loss.mean().backward()
		critic_loss.mean().backward()
		self.optim_actor.step()
		self.optim_critic.step()
