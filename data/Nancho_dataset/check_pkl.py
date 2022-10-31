import pickle
file = open('dtest.pkl', 'rb')

# dump information to that file
data = pickle.load(file)
print(data)
