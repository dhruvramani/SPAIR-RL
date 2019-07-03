import skvideo.io
from PIL import Image
import numpy as np
import sys
import gym
from PIL import Image
import numpy as np
import pickle
from random import randint


data_path ='../MontezumaRevenge/'
environment = 'MontezumaRevenge-v0'


#np.random.seed(seed=0)

random_episodes = 1000
dqn_episodes=100
steps= 20
img_h=64
img_w=64
ratio=1/10

rand_attempts =2

def remove_black_imgs(frames):
    corrupt_indices = []

    for i in range(frames.shape[0]):
        if (np.count_nonzero(frames[i, ...])   < img_h * img_w * ratio*3):
            corrupt_indices.append(i)

    frames = np.delete(frames, corrupt_indices, axis=0)
    return frames

def generate_data_dqn(data_type,size,max_size):

    print('generating dqn data',data_type)
    idx = 0

    frames = skvideo.io.vread(data_path+'dqn_vids/openaigym.video.0.8352.video{0:06d}'.format(0) + '.mp4')#[0::15, ...]
    print(type(frames))
    raise

    for episode in range(1,int(dqn_episodes*size)):
        print(episode)
        tmp = skvideo.io.vread(data_path+'dqn_vids/openaigym.video.0.8352.video{0:06d}'.format(episode) + '.mp4')[0::15, ...]
        frames = np.concatenate([frames,tmp], axis=0)

        print(frames.shape)

    frames_cropped = np.zeros((frames.shape[0]*2,img_h,img_w,frames.shape[-1]),dtype ='uint8')

    rand_coordinates = np.random.rand(frames.shape[0],2)
    rand_coordinates[:,0]=rand_coordinates[:,0]*(frames.shape[1]-img_h)
    rand_coordinates[:,1]=rand_coordinates[:,1]*(frames.shape[2]-img_w)
    rand_coordinates=rand_coordinates.astype(int)

    for i in range(frames.shape[0]):
        frames_cropped[i,...] = frames[i,rand_coordinates[i,0]:rand_coordinates[i,0]+img_h,rand_coordinates[i,1]:rand_coordinates[i,1]+img_w,:]


    rand_coordinates = np.random.rand(frames.shape[0],2)
    rand_coordinates[:,0]=rand_coordinates[:,0]*(frames.shape[1]-img_h)
    rand_coordinates[:,1]=rand_coordinates[:,1]*(frames.shape[2]-img_w)
    rand_coordinates=rand_coordinates.astype(int)


    for i in range(frames.shape[0]):
        frames_cropped[i+frames.shape[0],...] = frames[i,rand_coordinates[i,0]:rand_coordinates[i,0]+img_h,rand_coordinates[i,1]:rand_coordinates[i,1]+img_w,:]


    frames_cropped = remove_black_imgs(frames_cropped)

    frames_cropped = frames_cropped[:max_size,...]

    for i in range(frames_cropped.shape[0]):
        idx =write_to_im(frames_cropped,idx,data_path + 'dqn_imgs/' + data_type + '/' + str(episode) + '-' + str(0) + '_' + str( idx) + '.png')

    return frames_cropped


def generate_data_random(data_type,size,max_size):

    final_images = np.zeros((max_size,img_h,img_w,3))

    print('generating random data',data_type)
    idx = 0
    env = gym.make(environment)
    for episode in range(int(random_episodes*size)):
        print('episode: ',episode,'/',int(random_episodes*size))
        observation = env.reset()

        if(idx+(rand_attempts*2*steps)>=max_size):
            break

        for i in range(rand_attempts):
            rand_w = randint(0, np.shape(observation)[1] - img_w)
            rand_h = randint(0, np.shape(observation)[0] - img_h)
            observation = observation[rand_h:rand_h + img_h, rand_w:rand_w + img_w, :]

            if (np.count_nonzero(observation) >= img_h * img_w * ratio*3):
                final_images[idx,...]=observation
                idx = write_to_im(observation,idx,data_path + 'random_imgs/' + data_type + '/' + str(episode) + '-' + str(0) + '_'+str(idx)+'.png',random=True)

        for t in range(1,steps):
            action = env.action_space.sample()

            observation, reward, done, info = env.step(action)

            for i in range(rand_attempts):
                rand_w = randint(0, np.shape(observation)[1] - img_w)
                rand_h = randint(0, np.shape(observation)[0] - img_h)
                observation = observation[rand_h:rand_h + img_h, rand_w:rand_w + img_w, :]

                if done:
                    print("Episode finished after {} timesteps".format(t + 1))
                    break

                if (np.count_nonzero(observation) >= img_h * img_w * ratio*3):
                    final_images[idx, ...] = observation
                    idx = write_to_im(observation, idx,data_path + 'random_imgs/' + data_type + '/' + str(episode) + '-' + str(0) + '_' + str( idx) + '.png',random=True)

    env.close()

    return final_images


def write_to_im(image,idx,filename,random=False):

    if(random):
        array = image
    else:
        array  = image[idx, ...]
    im = Image.fromarray(array)
    im.save(filename)
    idx+=1
    return idx


if __name__ == '__main__':
    train_dqn = generate_data_dqn('train',.8,max_size=30000)
    train_random = generate_data_random('train',.8,max_size=10000)
    print(np.shape(train_dqn))
    print(np.shape(train_random))
    raise

    np.save(data_path + 'train', np.concatenate([train_random,train_dqn],axis=0))

    validation_random =generate_data_dqn('validation',.2,max_size=7500)
    validation_dqn = generate_data_random('validation',.2,max_size=2500)
    np.save(data_path + 'validation', np.concatenate([validation_random,validation_dqn],axis=0))

    test_random = generate_data_dqn('test',.2,max_size=7500)
    test_dqn = generate_data_random('test',.2,max_size=2500)
    np.save(data_path + 'test', np.concatenate([test_random,test_dqn],axis=0))


