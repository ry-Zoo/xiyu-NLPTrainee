class Solution(object):
    def partition(self, s):
        if len(s) == 0:
            return []
        else:
            res = []
            self.dividedAndsel(s, [], res)
        return res

    def dividedAndsel(self, s, tmp, res):
        if len(s) == 0:
            res.append(tmp)
        for i in range(1, len(s)+1):
            if s[:i] == s[:i][::-1]:
                self.dividedAndsel(s[i:], tmp + [s[:i]], res)
