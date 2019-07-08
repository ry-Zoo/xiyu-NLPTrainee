

raw_str=input()

word_generate=raw_str.split(" ")
word_lst=[i for i in word_generate if i!=" "]

reverse_word_lst=word_lst[::-1]

res_str=" ".join(reverse_word_lst)

print(res_str)