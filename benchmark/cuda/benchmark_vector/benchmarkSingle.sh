mem=$1
size=$2
slice=$(( 50000000 / $size ))
iter=$(( mem / $slice ))

for i in $(seq 1 $iter)
do
		amount=$(( amount + $slice ))
		for j in {1..5}
		do
			./BenchmarkCuda $amount 
		done
done
