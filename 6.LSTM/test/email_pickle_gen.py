account = {'email':'@naver.com', 'pw':'@'}
f = open('../../../../email_info.bin', 'wb')
import pickle
pickle.dump(account, f)
f.close()

# account = {"LINK_AWSAccessKeyId":"","LINK_AWSSecretKey":""}
# f = open('../../../../link_aws_key.bin', 'wb')
# import pickle
# pickle.dump(account, f)
# f.close()
