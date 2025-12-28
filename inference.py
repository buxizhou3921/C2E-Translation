"""train"""
# python src/train.py -model gru_seq2seq -train teacher
# python src/train.py -model gru_seq2seq -train free

# python src/train.py -model gru_attention -train teacher -align dot
# python src/train.py -model gru_attention -train teacher -align mul
# python src/train.py -model gru_attention -train teacher -align add
# python src/train.py -model gru_attention -train free -align dot
# python src/train.py -model gru_attention -train free -align mul
# python src/train.py -model gru_attention -train free -align add

# python src/train.py -model transformer -position absolute -norm layer
# python src/train.py -model transformer -position absolute -norm rms
# python src/train.py -model transformer -position relative -norm layer
# python src/train.py -model transformer -position relative -norm rms

# python src/train.py -model transformer -position absolute -norm layer -bs 64 -lr 5e-4 -heads 2 -layers 2
# python src/train.py -model transformer -position absolute -norm layer -bs 64 -lr 5e-4 -heads 2 -layers 4
# python src/train.py -model transformer -position absolute -norm layer -bs 64 -lr 5e-4 -heads 4 -layers 4

# python src/train.py -model transformer -position absolute -norm layer -bs 64 -lr 1e-3 -heads 2 -layers 2
# python src/train.py -model transformer -position absolute -norm layer -bs 64 -lr 1e-3 -heads 2 -layers 4
# python src/train.py -model transformer -position absolute -norm layer -bs 64 -lr 1e-3 -heads 4 -layers 4

# python src/train.py -model transformer -position absolute -norm layer -bs 128 -lr 1e-3 -heads 2 -layers 2
# python src/train.py -model transformer -position absolute -norm layer -bs 128 -lr 1e-3 -heads 4 -layers 4





"""evaluate"""
# python src/evaluate.py -model gru_seq2seq -train teacher -decode greedy/beam-search
# python src/evaluate.py -model gru_seq2seq -train free -decode greedy/beam-search

# python src/evaluate.py -model gru_attention -train teacher -align dot -decode greedy/beam-search
# python src/evaluate.py -model gru_attention -train teacher -align mul -decode greedy/beam-search
# python src/evaluate.py -model gru_attention -train teacher -align add -decode greedy/beam-search
# python src/evaluate.py -model gru_attention -train free -align dot -decode greedy/beam-search
# python src/evaluate.py -model gru_attention -train free -align mul -decode greedy/beam-search
# python src/evaluate.py -model gru_attention -train free -align add -decode greedy/beam-search

