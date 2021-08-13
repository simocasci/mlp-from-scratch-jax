import jax
import jax.numpy as jnp
from jax import grad, jit
from sklearn.utils import shuffle as shuff

class MLP:
	def __init__(self, layers, task):
		self.layers = layers
		assert(task in ['regression', 'classification'])
		self.task = task
		self.params = MLP.init_params(self.layers)
	
	@staticmethod
	def init_params(layers):
		key = jax.random.PRNGKey(0)
		initializer = jax.nn.initializers.he_normal()
		return [(initializer(key, shape=(i,j)),jax.random.uniform(key, (j,)))
				 for (i,j) in zip(layers[:-1],layers[1:])]
				 
	@staticmethod
	@jit
	def classify(params, xs):
		pred = xs
		for w,b in params[:-1]:
			pred = jax.nn.relu(jnp.dot(pred,w)+b)
		last_w,last_b = params[-1]
		return jax.nn.softmax(jnp.dot(pred,last_w)+last_b,axis=1)
		
	@staticmethod
	@jit
	def predict(params, xs):
		pred = xs
		for w,b in params[:-1]:
			pred = jax.nn.relu(jnp.dot(pred,w)+b)
		last_w,last_b = params[-1]
		return jnp.dot(pred,last_w)+last_b
		
	@staticmethod
	@jit
	def crossentropy(y_true, y_pred):
		return -jnp.sum(jnp.sum(y_true*jnp.log(y_pred+1e-9),axis=1))
		
	@staticmethod
	@jit
	def mse(y_true, y_pred):
		return jnp.mean(jnp.square(jnp.subtract(y_true,y_pred)))
		
	@jit
	def l2_loss(params, weight_decay):
		return weight_decay * sum([jnp.sum(jnp.square(l)) for l in jax.tree_leaves(params)])
		
	@staticmethod
	@jit
	def classification_loss(params, xs, ys, weight_decay):
		ys_hat = MLP.classify(params,xs)
		return MLP.crossentropy(ys,ys_hat) + MLP.l2_loss(params,weight_decay)
		
	@staticmethod
	@jit
	def regression_loss(params, xs, ys, weight_decay):
		ys_hat = MLP.predict(params,xs)
		return MLP.mse(ys,ys_hat) + MLP.l2_loss(params,weight_decay)
		
	@staticmethod
	@jit
	def classification_update(params, xs, ys, learning_rate, weight_decay):
		grads = grad(MLP.classification_loss)(params,xs,ys,weight_decay)
		return [(w-learning_rate*dw, b-learning_rate*db) for (w,b), (dw,db) in zip(params,grads)]
		
	@staticmethod
	@jit
	def regression_update(params, xs, ys, learning_rate, weight_decay):
		grads = grad(MLP.regression_loss)(params,xs,ys,weight_decay)
		return [(w-learning_rate*dw, b-learning_rate*db) for (w,b), (dw,db) in zip(params,grads)]
	
	def train(self, xs, ys, epochs=10, batch_size=128, learning_rate=0.001, weight_decay=0.01, shuffle=True):
		assert(len(xs) == len(ys))
		step_size = min(batch_size,len(ys))
		for _ in range(epochs):
			if shuffle:
				xs,ys = shuff(xs,ys,random_state=0)
			for i in range(0, len(ys), step_size):
				if self.task == 'classification':
					self.params = MLP.classification_update(self.params,xs[i:i+step_size],ys[i:i+step_size],learning_rate,weight_decay)
				else:
					 self.params = MLP.regression_update(self.params,xs[i:i+step_size],ys[i:i+step_size],learning_rate,weight_decay)	 
				
	def accuracy(self, xs, ys):
		if self.task == 'classification':
			labels = jnp.argmax(ys,axis=1)
			ys_hat = jnp.argmax(MLP.classify(self.params,xs),axis=1)
			return jnp.mean(ys_hat==labels)
		else:
			raise Exception("trying to calculate accuracy of a regression model, use mean_error(xs,ys) instead")
	
	def mean_squared_error(self, xs, ys):
		preds = MLP.predict(self.params,xs)
		return jnp.mean(jnp.square(ys-preds))