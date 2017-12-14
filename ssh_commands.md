Create ssh key:
ssh-keygen -t rsa -b 4096

View keys:
cd ~/.ssh
cat id_rsa
cat id_rsa.pub
pbcopy (type things, then paste normal way)
cat id_rsa.pub | pbcopy
pbpaste

ssh ubuntu@gpu-big-box.self-loop.com
scp cifar.py ubuntu@gpu-big-box.self-loop.com:
python cifar.py
