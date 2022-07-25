#!/bin/bash
make clean
make -j

# ------------------------------------------------------------------------------
#  Parameters for Point-to-Hyperplane Nearest Neighbor Search
# ------------------------------------------------------------------------------
dname=ImageNet
n=2340373
d=151
qn=100
b=0.9
cf=config
dtype=float32
dpath=../data/bin/${dname}
opath=../results/${dname}/

# ------------------------------------------------------------------------------
#  Competitors
# ------------------------------------------------------------------------------
# # Ground Truth
# ./p2h -alg 0 -n ${n} -qn ${qn} -d ${d} -dt ${dtype} -dn ${dname} \
#   -ds ${dpath}.ds -qs ${dpath}.q -ts ${dpath}.gt -op ${opath}

# # Linear-Scan
# ./p2h -alg 1 -n ${n} -qn ${qn} -d ${d} -dt ${dtype} -dn ${dname} \
#   -ds ${dpath}.ds -qs ${dpath}.q -ts ${dpath}.gt -op ${opath}

# # Ball-Tree
# for leaf in 100 200 500 1000 2000 5000 10000
# do 
#   ./p2h -alg 2 -n ${n} -qn ${qn} -d ${d} -leaf ${leaf} -cf ${cf} -dt ${dtype} \
#     -dn ${dname} -ds ${dpath}.ds -qs ${dpath}.q -ts ${dpath}.gt -op ${opath}
# done

# # BC-Tree
# for leaf in 100 200 500 1000 2000 5000 10000
# do 
#   ./p2h -alg 3 -n ${n} -qn ${qn} -d ${d} -leaf ${leaf} -cf ${cf} -dt ${dtype} \
#     -dn ${dname} -ds ${dpath}.ds -qs ${dpath}.q -ts ${dpath}.gt -op ${opath}
# done

# # FH
# for m in 128 # 8 16 32 64 128
# do 
#   for s in 1 2 4 8
#   do
#     ./p2h -alg 4 -n ${n} -qn ${qn} -d ${d} -m ${m} -s ${s} -b ${b} -cf ${cf} \
#       -dt ${dtype} -dn ${dname} -ds ${dpath}.ds -qs ${dpath}.q \
#       -ts ${dpath}.gt -op ${opath}
#   done
# done

# # NH (NH with LCCS-LSH)
# w=0.1
# for m in 8 16 32 64 128
# do
#   for s in 1 2 4 8
#   do
#     ./p2h -alg 5 -n ${n} -qn ${qn} -d ${d} -m ${m} -s ${s} -w ${w} -cf ${cf} \
#       -dt ${dtype} -dn ${dname} -ds ${dpath}.ds -qs ${dpath}.q \
#       -ts ${dpath}.gt -op ${opath}
#   done
# done

# # NH_DCF (NH with QALSH)
# for m in 8 16 32 64 128
# do
#   for s in 1 2 4 8
#   do
#     ./p2h -alg 6 -n ${n} -qn ${qn} -d ${d} -m ${m} -s ${s} -cf ${cf} \
#       -dt ${dtype} -dn ${dname} -ds ${dpath}.ds -qs ${dpath}.q \
#       -ts ${dpath}.gt -op ${opath}
#   done
# done

# # Ortho_LSH
# for m in 8 16 32 64 128 256
# do
#   ./p2h -alg 7 -n ${n} -qn ${qn} -d ${d} -m ${m} -cf ${cf} -dt ${dtype} \
#     -dn ${dname} -ds ${dpath}.ds -qs ${dpath}.q -ts ${dpath}.gt -op ${opath}
# done

# # Metric_Tree
# for leaf in 0.01 0.03 0.05
# do 
#   for m in 8 16 32 64 128
#   do
#     ./p2h -alg 8 -n ${n} -qn ${qn} -d ${d} -m ${m} -l ${leaf} -cf ${cf} -dt ${dtype} \
#       -dn ${dname} -ds ${dpath}.ds -qs ${dpath}.q -ts ${dpath}.gt -op ${opath}
#   done 
# done 
