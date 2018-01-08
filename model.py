from __future__ import division
import os
import time
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange
from collections import namedtuple
from tqdm import trange
from fr import *
from collections import deque
from module import *
from utils import *
# model, train, test

def slerp(val, low, high): # Spherical linear interpolation
    # Used to achieve interpolations of images in latent space
    omega = np.arccos(np.clip(np.dot(low/np.linalg.norm(low),high/np.linalg.norm(high)),-1,1) )
    so = np.sin(omega)
    if so == 0:
        return (1.0-val) * low + val*high
    return np.sin((1.0-val) * omega)/so * low + np.sin(val * omega) / so * high

class frgan(object):
    def __init__(self, args):
        self.is_train= args.phase
        self.batch_size = args.batch_size
        self.scale_size = args.scale_size
        self.discriminator = discriminator
        self.generator = generator
        self.data_loader = get_loader(self.batch_size,scale_size=self.scale_size)
        self.z_num = args.z_num
        self.repeat_num = int(np.log2(self.scale_size))-2
        self.hidden_num = args.hidden_num
        self.optimizer = 'adam'

        self.step = tf.Variable(0,name='step',trainable=False)
        self.beta1 = args.beta1
        self.beta2 = args.beta2
        self.g_lr = tf.Variable(args.g_lr,name='g_lr')
        self.d_lr = tf.Variable(args.d_lr, name='g_lr')

        self.g_lr_update = tf.assign(self.g_lr, tf.maximum(self.g_lr * 0.5, args.lr_lower_boundary),
                                     name = 'g_lr_update')
        self.d_lr_update = tf.assign(self.d_lr, tf.maximum(self.d_lr * 0.5, args.lr_lower_boundary),
                                     name= 'd_lr_update')
        self.gamma = args.gamma
        self.lambda_k = args.lambda_k
        self.low_rate = args.low_rate
        self.high_rate = args.high_rate
        self.start_step = 0
        self.log_step = args.log_step
        self.max_step = args.max_step
        self.lr_update_step = args.lr_update_step
        self.model_dir = args.logs_dir
        self.test_dir = args.test_dir
        self.build_model()
        self.saver = tf.train.Saver()
        self.summary_writer = tf.summary.FileWriter(self.model_dir)
        sv = tf.train.Supervisor(
                                 logdir = self.model_dir,
                                 is_chief= True,
                                 saver = self.saver,
                                 summary_op= None,
                                 summary_writer= self.summary_writer,
                                 save_model_secs= 300,
                                 global_step = self.step,
                                 ready_for_local_init_op= None
                                 )
        gpu_options = tf.GPUOptions(allow_growth= True)
        sess_config = tf.ConfigProto(allow_soft_placement= True,
                                     gpu_options= gpu_options)

        self.sess= sv.prepare_or_wait_for_session(config=sess_config)

        if self.is_train!='train':
            g =tf.get_default_graph()
            g._finalized = False

            self.build_test_model()

    def get_image_from_loader(self):
        x = self.data_loader.eval(session=self.sess)
        return x


    def build_model(self):
        self.x = self.data_loader
        x = norm_img(self.x)

        self.z = tf.random_uniform(
            (tf.shape(x)[0], self.z_num), minval=-1.0, maxval=1.0)
        self.k_t0 = tf.Variable(0., trainable=False, name='k_t0')
        self.k_t1 = tf.Variable(0., trainable=False, name='k_t1')

        self.G0, self.G_var = self.generator(
            self.z, self.hidden_num,
            self.repeat_num, reuse=False)

        G_fft0 = fft(self.G0,scale_size1=0,scale_size2=15, shape=[self.scale_size,self.scale_size])
        x_fft0 = self.low_rate * fft(x, scale_size1=0, scale_size2=15, shape=[self.scale_size, self.scale_size])
        G_fft1 = fft(self.G0, scale_size1=15, scale_size2=-1, shape=[self.scale_size, self.scale_size])
        x_fft1 = self.high_rate * fft(x, scale_size1=15, scale_size2=-1, shape=[self.scale_size, self.scale_size])


        d_out0, self.D_z0, self.D_var0 = self.discriminator(
            tf.concat([G_fft0, x_fft0], 0), self.z_num,
            self.hidden_num, self.repeat_num,name="D0")
        d_out1, self.D_z1, self.D_var1 = self.discriminator(
            tf.concat([G_fft1, x_fft1], 1), self.z_num,
            self.hidden_num, self.repeat_num, name="D1")
        AE_G0, AE_x0 = tf.split(d_out0, 2)
        AE_G1, AE_x1 = tf.split(d_out1, 2)

        self.G = denorm_img_nc(self.G0)
        self.G_fft0, self.x_fft0 = denorm_img_nc(G_fft0), denorm_img_nc(x_fft0)
        self.G_fft1, self.x_fft1 = denorm_img_nc(G_fft1), denorm_img_nc(x_fft1)
        self.AE_G0, self.AE_x0 = denorm_img_nc(AE_G0), denorm_img_nc(AE_x0)
        self.AE_G1, self.AE_x1 = denorm_img_nc(AE_G1), denorm_img_nc(AE_x1)

        optimizer = tf.train.AdamOptimizer

        g_optimizer0, d_optimizer0 = optimizer(self.g_lr), optimizer(self.d_lr)
        g_optimizer1, d_optimizer1 = optimizer(self.g_lr), optimizer(self.d_lr)

        self.d_loss_real0 = tf.reduce_mean(tf.abs(AE_x0 - x_fft0))
        self.d_loss_fake0 = tf.reduce_mean(tf.abs(AE_G0 - G_fft0))
        self.d_loss_real1 = tf.reduce_mean(tf.abs(AE_x1 - x_fft1))
        self.d_loss_fake1 = tf.reduce_mean(tf.abs(AE_G1 - G_fft1))

        self.d_loss_real = self.d_loss_real0+self.d_loss_real1
        self.d_loss_fake = self.d_loss_fake0+self.d_loss_fake1

        self.d_loss0 = self.d_loss_real0 - self.k_t0 * self.d_loss_fake0
        self.g_loss0 = self.d_loss_fake0

        self.d_loss1 = self.d_loss_real1 - self.k_t1 * self.d_loss_fake1
        self.g_loss1 = self.d_loss_fake1

        self.d_loss = self.d_loss0 + self.d_loss1
        self.g_loss = self.g_loss0 + self.g_loss1

        d_optim0 = d_optimizer0.minimize(self.d_loss0, var_list=self.D_var0)
        g_optim0 = g_optimizer0.minimize(self.g_loss0, global_step=self.step, var_list=self.G_var)
        d_optim1 = d_optimizer1.minimize(self.d_loss1, var_list=self.D_var1)
        g_optim1 = g_optimizer1.minimize(self.g_loss1, global_step=self.step, var_list=self.G_var)

        self.balance0 = self.gamma * self.d_loss_real0 - self.g_loss0
        self.measure0 = self.d_loss_real0 + tf.abs(self.balance0)

        self.balance1 = self.gamma * self.d_loss_real1 - self.g_loss1
        self.measure1 = self.d_loss_real1 + tf.abs(self.balance1)

        self.balance = self.balance0 +self.balance1
        self.measure = self.measure0 + self.measure1

        with tf.control_dependencies([d_optim0, g_optim0]):
            self.k_update0 = tf.assign(
                self.k_t0, tf.clip_by_value(self.k_t0 + self.lambda_k * self.balance0, 0, 1))
        with tf.control_dependencies([d_optim1, g_optim1]):
            self.k_update1 = tf.assign(
                self.k_t1, tf.clip_by_value(self.k_t1 + self.lambda_k * self.balance1, 0, 1))

        self.summary_op = tf.summary.merge([
            tf.summary.image("G", self.G),
            tf.summary.image("G_fft0", self.G_fft0),
            tf.summary.image("x_fft0", self.x_fft0),
            tf.summary.image("G_fft1", self.G_fft1),
            tf.summary.image("x_fft1", self.x_fft1),
            tf.summary.image("AE_G", self.AE_G0),
            tf.summary.image("AE_x", self.AE_x0),
            tf.summary.image("AE_G", self.AE_G1),
            tf.summary.image("AE_x", self.AE_x1),

            tf.summary.scalar("loss/d_loss0", self.d_loss0),
            tf.summary.scalar("loss/d_loss_real0", self.d_loss_real0),
            tf.summary.scalar("loss/d_loss_fake0", self.d_loss_fake0),
            tf.summary.scalar("loss/g_loss0", self.g_loss0),
            tf.summary.scalar("misc/measure0", self.measure0),
            tf.summary.scalar("misc/k_t0", self.k_t0),
            tf.summary.scalar("misc/balance0", self.balance0),

            tf.summary.scalar("loss/d_loss1", self.d_loss1),
            tf.summary.scalar("loss/d_loss_real1", self.d_loss_real1),
            tf.summary.scalar("loss/d_loss_fake1", self.d_loss_fake1),
            tf.summary.scalar("loss/g_loss1", self.g_loss1),
            tf.summary.scalar("misc/measure1", self.measure1),
            tf.summary.scalar("misc/k_t1", self.k_t1),
            tf.summary.scalar("misc/balance1", self.balance1),

            tf.summary.scalar("loss/d_loss", self.d_loss),
            tf.summary.scalar("loss/d_loss_real", self.d_loss_real),
            tf.summary.scalar("loss/d_loss_fake", self.d_loss_fake),
            tf.summary.scalar("loss/g_loss", self.g_loss),
            tf.summary.scalar("misc/measure", self.measure),
            tf.summary.scalar("misc/d_lr", self.d_lr),
            tf.summary.scalar("misc/g_lr", self.g_lr),
            tf.summary.scalar("misc/balance", self.balance),
        ])
    def train(self):

        z_fixed = np.random.uniform(-1,1, size=(self.batch_size, self.z_num))
        x_fixed = self.get_image_from_loader()
        save_image(x_fixed,'{}/x_fixed.png'.format(self.model_dir))
        measure_history =deque([0]*self.lr_update_step, self.lr_update_step)

        for step in trange(self.start_step, self.max_step):
            fetch_dict = {
                "k_update0": self.k_update0,
                "k_update1": self.k_update1,
                "measure0": self.measure0,
                "measure1": self.measure1,
                "measure": self.measure,

            }

            if step % self.log_step ==0:
                fetch_dict.update({
                    "summary": self.summary_op,
                    "g_loss": self.g_loss,
                    "d_loss": self.d_loss,
                    "g_loss0": self.g_loss0,
                    "d_loss0": self.d_loss0,
                    "g_loss1": self.g_loss1,
                    "d_loss1": self.d_loss1,
                    "k_t0": self.k_t0,
                    "k_t1": self.k_t1,
                })
            result = self.sess.run(fetch_dict)

            measure = result['measure']
            measure0 = result['measure0']
            measure1 = result['measure1']
            measure_history.append(measure)


            if step % self.log_step == 0:
                self.summary_writer.add_summary(result['summary'],step)
                self.summary_writer.flush()

                g_loss = result['g_loss']
                d_loss = result['d_loss']
                g_loss0 = result['g_loss0']
                d_loss0 = result['d_loss0']
                k_t0 = result['k_t0']
                g_loss1 = result['g_loss1']
                d_loss1 = result['d_loss1']
                k_t1 = result['k_t1']

                print("[{}/{}] Loss_D: {:.6f} Loss_G: {:.6f} measure: {:.4f}"
                      "Loss_D0: {:.6f} Loss_G0: {:.6f} measure0: {:.4f} k_t0:{:4f} "
                      "Loss_D1: {:.6f} Loss_G1: {:.6f} measure1: {:.4f} k_t1:{:4f}"
                      .format(step,self.max_step, d_loss, g_loss, measure,
                              d_loss0, g_loss0, measure0, k_t0,d_loss1, g_loss1, measure1, k_t1))

            if step % (self.log_step * 10) == 0:
                x_fake,_ = self.generate(z_fixed, self.model_dir, idx=step)
                self.autoencode(x_fixed, self.model_dir,idx=step, x_fake=x_fake)

            if step % self.lr_update_step == self.lr_update_step - 1:
                self.sess.run([self.g_lr_update, self.d_lr_update])

    def build_test_model(self):
        with tf.variable_scope("test") as vs:
            z_optimizer = tf.train.AdamOptimizer(0.0001)

            self.z_r = tf.get_variable("z_r",[self.batch_size,self.z_num],tf.float32)
            self.z_r_update = tf.assign(self.z_r,self.z)
        G_z_r, _ = self.generator(self.z_r, self.hidden_num, self.repeat_num,reuse = True)

        with tf.variable_scope("test") as vs:
            self.z_r_loss = tf.reduce_mean(tf.abs(self.x-G_z_r))
            self.z_r_optim = z_optimizer.minimize(self.z_r_loss, var_list=[self.z_r])

        test_variables = tf.contrib.framework.get_variables(vs)
        self.sess.run(tf.variables_initializer(test_variables))

    def generate(self,inputs, root_path= None, path= None,idx=None, save= True):
        G = self.sess.run(self.G, {self.z: inputs})
        G_fft0 = self.sess.run(self.G_fft0, {self.z: inputs})
        G_fft1 = self.sess.run(self.G_fft1, {self.z: inputs})
        G0 = self.sess.run(self.G0, {self.z: inputs})
        if path is None and save:
            path = os.path.join(root_path,'{}_G.png'.format(idx))
            save_image(G, path)
            print("[*]Sample saved: {}".format(path))
            path0 = os.path.join(root_path, '{}_G_fft0.png'.format(idx))
            save_image(G_fft0, path0)
            print("[*]Sample saved: {}".format(path0))
            path1 = os.path.join(root_path, '{}_G_fft1.png'.format(idx))
            save_image(G_fft1, path1)
            print("[*]Sample saved: {}".format(path1))
        return G0,G

    def autoencode(self,inputs, path, idx = None, x_fake= []):
        key0='real0'
        key1 = 'real1'
        key2 = 'fake0'
        key3 ='fake1'
        x_path0 = os.path.join(path,'{}_D_{}.png'.format(idx,key0))
        x0 = self.sess.run(self.AE_x0,{self.x: inputs})
        save_image(x0,x_path0)
        print("[*] Samples saved: {}".format(x_path0))
        x_path1 = os.path.join(path, '{}_D_{}.png'.format(idx, key1))
        x1 = self.sess.run(self.AE_x1, {self.x: inputs})
        save_image(x1, x_path1)
        print("[*] Samples saved: {}".format(x_path1))
        if  x_fake==[]:
            x_fake=[]
        else:
            G_path0 = os.path.join(path, '{}_D_{}.png'.format(idx, key2))
            G0 = self.sess.run(self.AE_G0, {self.G0:x_fake})
            save_image(G0, G_path0)
            print("[*] Samples saved: {}".format(G_path0))

            G_path1 = os.path.join(path, '{}_D_{}.png'.format(idx, key3))
            G1 = self.sess.run(self.AE_G1, {self.G0:x_fake})
            save_image(G1, G_path1)
            print("[*] Samples saved: {}".format(G_path1))

    def encode(self, inputs):

        return self.sess.run([self.D_z0,self.D_z1], {self.x: inputs})

    def decode(self,z0,z1):
        return self.sess.run([self.AE_x0,self.AE_x1],{self.D_z0:z0,self.D_z1:z1})

    def interpolate_G(self, real_batch, step=0, root_path='.', train_epoch=0):
        batch_size = len(real_batch)
        half_batch_size = int(batch_size / 2)

        self.sess.run(self.z_r_update)
        tf_real_batch = real_batch
        for i in trange(train_epoch):
            z_r_loss, _ = self.sess.run([self.z_r_loss, self.z_r_optim], {self.x: tf_real_batch})
        z = self.sess.run(self.z_r)

        z1, z2 = z[:half_batch_size], z[half_batch_size:]
        real1_batch, real2_batch = real_batch[:half_batch_size], real_batch[half_batch_size:]

        generated = []
        for idx, ratio in enumerate(np.linspace(0, 1, 10)):
            z = np.stack([slerp(ratio, r1, r2) for r1, r2 in zip(z1, z2)])
            _,z_decode = self.generate(z, save=False)
            generated.append(z_decode)

        generated = np.stack(generated).transpose([1, 0, 2, 3, 4])
        for idx, img in enumerate(generated):
            save_image(img, os.path.join(root_path, 'test{}_interp_G_{}.png'.format(step, idx)), nrow=10)

        all_img_num = np.prod(generated.shape[:2])
        batch_generated = np.reshape(generated, [all_img_num] + list(generated.shape[2:]))
        save_image(batch_generated, os.path.join(root_path, 'test{}_interp_G.png'.format(step)), nrow=10)


    def interpolate_D(self, real1_batch, real2_batch, step=0, root_path="."):
        [real1_encode0,real1_encode1] = self.encode(real1_batch)
        [real2_encode0,real2_encode1] = self.encode(real2_batch)
        [real1_batch_00, real1_batch_11] = self.sess.run([self.AE_x0,self.AE_x1], {self.x: real1_batch})
        [real2_batch_00, real2_batch_11] = self.sess.run([self.AE_x0,self.AE_x1], {self.x: real2_batch})
        decodes0 = []
        decodes1 = []
        for idx, ratio in enumerate(np.linspace(0, 1, 10)):
            z0 = np.stack([slerp(ratio, r1, r2) for r1, r2 in zip(real1_encode0, real2_encode0)])
            z1 = np.stack([slerp(ratio, r1, r2) for r1, r2 in zip(real1_encode1, real2_encode1)])
            [z_decode0,z_decode1] = self.decode(z0,z1)
            decodes0.append(z_decode0)
            decodes1.append(z_decode1)
        decodes0 = np.stack(decodes0).transpose([1, 0, 2, 3, 4])
        decodes1 = np.stack(decodes1).transpose([1, 0, 2, 3, 4])

        for idx, img in enumerate(decodes0):
            img = np.concatenate([[real1_batch[idx]],[real1_batch_00[idx]], img,
                                  [real2_batch_00[idx]],[real2_batch[idx]]], 0)
            save_image(img, os.path.join(root_path, 'test{}_interp_D0_{}.png'.format(step, idx)), nrow=10 + 4)

        for idx, img in enumerate(decodes1):
            img = np.concatenate([[real1_batch[idx]],[real1_batch_11[idx]],  img,
                                  [real2_batch_11[idx]],[real2_batch[idx]]], 0)
            save_image(img, os.path.join(root_path, 'test{}_interp_D1_{}.png'.format(step, idx)), nrow=10 + 4)

    def test(self):
        root_path = self.test_dir
        all_G_z = None
        for step in range(3):
            real1_batch = self.get_image_from_loader()
            real2_batch = self.get_image_from_loader()

            save_image(real1_batch, os.path.join(root_path, 'test{}_real1.png'.format(step)))
            save_image(real2_batch, os.path.join(root_path, 'test{}_real2.png'.format(step)))

            self.autoencode(real1_batch, "./", idx=os.path.join(root_path,"test{}_real1".format(step)))
            self.autoencode(real2_batch, "./", idx=os.path.join(root_path, "test{}_real2".format(step)))

            self.interpolate_G(real1_batch,step,root_path)
            self.interpolate_D(real1_batch, real2_batch, step, root_path)

            z_fixed = np.random.uniform(-1,1,size=(self.batch_size,self.z_num))
            _,G_z = self.generate(z_fixed, path=os.path.join(root_path,"test{}_G_z.png".format(step)))

            if all_G_z is None:
                all_G_z= G_z
            else:
                all_G_z = np.concatenate([all_G_z,G_z])
            # print(all_G_z.shape)
            save_image(all_G_z,'{}/G_z{}.png'.format(root_path,step))
        save_image(all_G_z,'{}/all_G_z.png'.format(root_path),nrow=16)


