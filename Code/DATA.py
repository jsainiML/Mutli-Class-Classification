# Creating multi-class data with 6 different cluster points with less than moderate deviation from each other
X_blob, y_blob = make_blobs(n_samples=1200,n_features=2, centers= 6, cluster_std= 1,random_state= 62)

# Numpy -> tensor/LongTensor
X_blob = torch.from_numpy(X_blob).type(torch.float)
y_blob = torch.from_numpy(y_blob).type(torch.LongTensor)

# Split Train/Test
Xb_train, Xb_test, yb_train, yb_test = train_test_split(X_blob, y_blob, test_size = 0.2,random_state=62)

# A quick verification on values
circle = pd.DataFrame({'X1': X_blob[:, 0], 'X2': X_blob[:,1], 'label' : y_blob})  # organizing the values in tabular format for better reading
circle.head(n=4)
