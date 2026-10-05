
from mask_utils import mask_text

test_text = """[Subreddit: addiction] Sugar and the blues . I found this link at r/nootropics , https://www.ncbi.nlm.nih.gov/pubmed/29191750 , it's about depression and anxiety brought on by sucrose withdrawal. I'm 2 years off booze and am in the process of attacking my 35 year long relationship with nicotine ( almost two weeks ). Now , about a month ago I stopped sugar and caffeine ( I was a two sugars in coffee guy - several times a day .) My thrust is that I had intuited ( remembered reading ... ) about the effects on mood that sugar buzzes and downers had . so , before the attack on the cigarette addiction I binned the sugar . I think I did notice a bit more calmness ... I have back slid for sure , but just today I cooked some healthy food and didn't go and buy a stack of 'cheer up ' confection . Most of you are probably well onto this , or may disagree , but I just wanted to say that I think sugar is a really underestimated mood swinging fuel. It seems to me that people can 'forget' about a seemingly innocent 'food' , and be banged around by it while they are battling the big bad drugs . Addiction is a demon . Good luck folks !"""

masked, lut = mask_text(test_text)
print("Original:", test_text)
print("\nMasked:", masked)
print("\nLookup table:", lut)
