
import numpy as np
import matplotlib as mpl
mpl.use("TKAgg")
import matplotlib.pyplot as plt

import sys
import torch
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from Part_1_MLP import MLP
import torch.optim as optim
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
  # Task setup block starts 
  # Do not change
  torch.manual_seed(1000)
  dataset = datasets.MNIST(
    root="dataset/", train=True, transform=transforms.ToTensor(), download=True)
  loader = DataLoader(dataset=dataset, batch_size=1, shuffle=True)
  # Task setup block end

  # Learner setup block
  seed = 0 if len(sys.argv) == 1 else int(sys.argv[1])
  torch.manual_seed(seed)  # do not change. This is for learners randomization
  ####### Start
  my_agent = MLP()
  my_agent.to(device)
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(my_agent.parameters(), lr=0.0003)
  rand_generator = np.random.RandomState(seed)
  batch_size = 100
  num_epochs = 10
  mini_batch_size = 20
  num_mini_batch = int(batch_size/mini_batch_size)
  image_batch = torch.zeros([batch_size,1,1,28,28], dtype=torch.float32, device=device)
  label_batch = torch.zeros([batch_size,1], dtype=torch.long, device=device)
  counter = 0
  epoches_shrink_ratio = 0.998
  epochs_num_list = []
  ####### End

  # Experiment block starts
  errors = []
  checkpoint = 1000
  correct_pred = 0
  for idx, (image, label) in enumerate(loader):
    # Observe
    label = label.to(device=device)
    image = image.to(device=device)

    # Make a prediction of label
    ####### Start
    # Replace the following statement with your own code for
    # making label prediction
    softmax_prob = my_agent.forward(image)
    softmax_prob = softmax_prob.cpu().detach().numpy() 
    pred_label = rand_generator.choice(list(range(10)), p=np.exp(softmax_prob[0]))
    ####### End

    # Evaluation
    correct_pred += (pred_label == label).sum()

    # Learn
    ####### Start
    
    image_batch[counter] = image
    label_batch[counter] = label
    counter +=1
    if counter == batch_size:
        counter = 0 #Reinitialize the counter
        if num_epochs == 1:
            pass
        else:
            num_epochs = num_epochs*epoches_shrink_ratio
        epochs_num_list.append(num_epochs)
        for e in range(round(num_epochs)):
            new_index = torch.randperm(batch_size)
            image_batch=image_batch[new_index]
            label_batch=label_batch[new_index]
            for m in range(num_mini_batch):
                current_image_mini_batch = image_batch[m*mini_batch_size:(m+1)*mini_batch_size]
                current_label_mini_batch = label_batch[m*mini_batch_size:(m+1)*mini_batch_size]
                loss = 0
                optimizer.zero_grad()
                #Now we add the loss for each samples in a mini-batch
                for sample_index in range(mini_batch_size):
                    output = my_agent(current_image_mini_batch[sample_index])
                    partial_loss = criterion(output, current_label_mini_batch[sample_index])
                    loss = loss+ partial_loss
                loss = loss/mini_batch_size
                loss.backward()
                optimizer.step()
        image_batch = torch.zeros([batch_size,1,1,28,28], dtype=torch.float32, device=device)
        label_batch = torch.zeros([batch_size,1], dtype=torch.long, device=device)
    ####### End

    # Log
    if (idx+1) % checkpoint == 0:
      error = float(correct_pred) / float(checkpoint) * 100
      print(error)
      errors.append(error)
      correct_pred = 0

      #plt.clf()
      #plt.plot(range(checkpoint, (idx+1)+checkpoint, checkpoint), errors)
      #plt.ylim([0, 100])
      #plt.pause(0.001)
  name = sys.argv[0].split('.')[-2].split('_')[-1]
  data = np.zeros((2, len(errors)))
  data[0] = range(checkpoint, 60000+1, checkpoint)
  data[1] = errors
  np.savetxt(name+str(seed)+".txt", data)
  plt.show()


if __name__ == "__main__":
  main()
