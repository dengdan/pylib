PYLIB=$(pwd)/../src

for SRC in ~/.bashrc ~/.zshrc
do
	echo $SRC
	echo "alias p='python'" >> $SRC
	echo "alias n='nvidia-smi'" >> $SRC
	echo "alias wn='watch nvidia-smi'" >> $SRC
	echo "export PYTHONPATH=$PYLIB:$PYTHONPATH" >> $SRC
done

