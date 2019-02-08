import pickle
f = open('../../../../email_info.bin', 'rb')
a = pickle.load(f)
print(a)
print(type(a))
print(a.keys())
print(a['email'])
print(a['pw'])

# import pickle
# f = open('../../../../link_aws_key.bin', 'rb')
# a = pickle.load(f)
# print(a)
# print(type(a))
# print(a.keys())
# print(a['LINK_AWSAccessKeyId'])
# print(a['LINK_AWSSecretKey'])