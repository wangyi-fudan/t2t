# t2t
LLM analogue of sed/awk, transform text line by line

make

sudo cp t2t /usr/bin/

mkdir /home/username/config/t2t

edit t2trc, change to your LLM API

cp t2trc /home/username/config/t2t/

cat gene.txt | t2t "Is the gene ralated to diabetes?"
