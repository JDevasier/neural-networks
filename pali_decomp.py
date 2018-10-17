
def main():

    input_str = "nitin"

    palindromic_decomp(0, len(input_str), input_str)

    print(all_part, curr_part)

all_part = []
curr_part = []

def isPalindrome(_str, low, hi):
    while (low < hi):
        if (_str[low] != _str[hi]):
            return False
        low += 1
        hi -= 1
    return True

def palindromic_decomp(start, n, _str):
    global all_part, curr_part
    if start >= n:
        print(start, n)
        all_part.append(curr_part)
        return

    for i in range(start, n):
        print(start, i)
        #print(i, _str[start:i-start+1])
        if isPalindrome(_str, start, i):
            print("palin:", _str[start:i-start+1] )
                
            curr_part.append(_str[start:i-start+1])

            palindromic_decomp(i+1, n, _str)

            curr_part.pop()

main()