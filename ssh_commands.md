scp cifar.py ubuntu@gpu-big-box.self-loop.com:

ssh-keygen -t rsa -b 4096
cd ~/.ssh
cat id_rsa
cat id_rsa.pub
pbcopy (type things, then paste normal way)
cat id_rsa.pub | pbcopy
pbpaste

ssh ubuntu@gpu-big-box.self-loop.com
then just run the file!
python cifar.py
