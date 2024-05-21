import os 

for i in range(10):
    os.system('python reg_pyt_src/main.py > reg_pyt_src/results/vgg/vgg_bn_base'+str(i+1)+'.txt')