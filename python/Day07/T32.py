def  is_substr(raw_str,enum_ls):
	enum_bk=enum_ls[:]
	if len(raw_str)==0 or len(enum_ls)==0:
		return False
	else:
		start=0
		tmp_ls=[]
		wflag=False
		sflag=False
		for s in range(0,len(raw_str)):
			for e in range(0,len(enum_ls)):
				if raw_str[s]==enum_ls[e][0]:
					start=s
					tmp_ls.append(raw_str[s])
					wflag=True
				else:
					if wflag==True:
						valid_str=raw_str[start:s+1]
						valid_str=valid_word(valid_str,enum_ls):
							enum_ls.remove(valid_str)
							sflag=True
						else:
							continue
					else:
						continue

def valid_word(valid_str,enum_ls):
	if valid_str in enum_ls:
		return valid_str



class Solution:
    def findSubstring(self, s: str, words: List[str]) -> List[int]:
        if not words:
            return []
        #wide:单词长度，num：单词个数，n：字符长度
        re, wide, num, n = list(), len(words[0]), len(words), len(s)
        long = wide * num
        if n < long:
            return []
        words.sort()
        #i表示长度为long的子串可能的开始节点
        for i in range(n+1-long):
            #直接将可能起始节点后置的元素切割，对比，如果和给定数组一致则输出起始节点
            l = [s[j:j+wide] for j in range(i, i+long, wide)]
            l.sort()
            if l == words:
                re.append(i)
        return re





