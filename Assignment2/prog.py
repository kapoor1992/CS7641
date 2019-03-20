import pca, ica, random_projection, factor_an, nn, exmax, kmeans
from utilities.data_fetcher import *
from utilities.data_transformer import *

print('kmeans orig mamm skin')
## KMEANS
# original clusters, 2, 2
train_count, X, y, test_attributes, test_labels = get_mammography_data(100)
kmeans.run(X)
train_count, X, y, test_attributes, test_labels = get_skin_data(10)
kmeans.run(X)

print('pca mamm skin')
# dimensionality finder and new clusters, 7, 2
pca.run_mamm()
pca.run_skin()
print('pca kmeans mamm skin')
tc, X, X_test, y_train, y_test = pca.get_mamm()
kmeans.run(X)
tc, X, X_test, y_train, y_test = pca.get_skin()
kmeans.run(X)

print('ica mamm skin')
# dimensionality finder and new clusters, 2, 3
ica.run_mamm()
ica.run_skin()
print('ica kmeans mamm skin')
tc, X, X_test, y_train, y_test = ica.get_mamm()
kmeans.run(X)
tc, X, X_test, y_train, y_test = ica.get_skin()
kmeans.run(X)

print('rp mamm skin')
# dimensionality finder and new clusters, 2, 2
random_projection.run_mamm()
random_projection.run_skin()
print('rp kmeans mamm skin')
tc, X, X_test, y_train, y_test = random_projection.get_mamm()
kmeans.run(X)
tc, X, X_test, y_train, y_test = random_projection.get_skin()
kmeans.run(X)

print('fa mamm skin')
# dimensionality finder and new clusters, 7, 2
factor_an.run_mamm()
factor_an.run_skin()
print('fa kmeans mamm skin')
tc, X, X_test, y_train, y_test = factor_an.get_mamm()
kmeans.run(X)
tc, X, X_test, y_train, y_test = factor_an.get_skin()
kmeans.run(X)

print('em orig mamm skin')
##EXMAX STUFF
# original clusters, 2, 4
train_count, X, y, test_attributes, test_labels = get_mammography_data(100)
exmax.run(X)
train_count, X, y, test_attributes, test_labels = get_skin_data(10)
exmax.run(X)

print('pca mamm skin')
# dimensionality finder and new clusters, 7, 2
pca.run_mamm()
pca.run_skin()
print('pca em mamm skin')
tc, X, X_test, y_train, y_test = pca.get_mamm()
exmax.run(X)
tc, X, X_test, y_train, y_test = pca.get_skin()
exmax.run(X)

print('ica mamm skin')
# dimensionality finder and new clusters, 6, 3
ica.run_mamm()
ica.run_skin()
print('ica em mamm skin')
tc, X, X_test, y_train, y_test = ica.get_mamm()
exmax.run(X)
tc, X, X_test, y_train, y_test = ica.get_skin()
exmax.run(X)

print('rp mamm skin')
# dimensionality finder and new clusters, 7, 5
random_projection.run_mamm()
random_projection.run_skin()
print('rp em mamm skin')
tc, X, X_test, y_train, y_test = random_projection.get_mamm()
exmax.run(X)
tc, X, X_test, y_train, y_test = random_projection.get_skin()
exmax.run(X)

print('fa mamm skin')
# dimensionality finder and new clusters, 7, 2
factor_an.run_mamm()
factor_an.run_skin()
print('fa em mamm skin')
tc, X, X_test, y_train, y_test = factor_an.get_mamm()
exmax.run(X)
tc, X, X_test, y_train, y_test = factor_an.get_skin()
exmax.run(X)

## NEURAL NET STUFF
train_percentages = [20, 40, 60, 80]

print('orig nn')
#orig nn
nn.run_mamm_orig(train_percentages)

print('pca nn mamm')
#dimension reduced nn
for train_percentage in train_percentages:
    train_count, train_attributes, train_labels, test_attributes, test_labels = pca.get_mamm(train_percentage)
    nn.run(train_attributes, train_labels, test_attributes, test_labels, train_percentage)

print('ica nn mamm')
#dimension reduced nn
for train_percentage in train_percentages:
    train_count, train_attributes, train_labels, test_attributes, test_labels = ica.get_mamm(train_percentage)
    nn.run(train_attributes, train_labels, test_attributes, test_labels, train_percentage)

print('rp nn mamm')
#dimension reduced nn
for train_percentage in train_percentages:
    train_count, train_attributes, train_labels, test_attributes, test_labels = random_projection.get_mamm(train_percentage)
    nn.run(train_attributes, train_labels, test_attributes, test_labels, train_percentage)

print('fa nn mamm')
#dimension reduced nn
for train_percentage in train_percentages:
    train_count, train_attributes, train_labels, test_attributes, test_labels = factor_an.get_mamm(train_percentage)
    nn.run(train_attributes, train_labels, test_attributes, test_labels, train_percentage)


'''
## REDUCED CLUSTER FINDER
tc, X, X_test, y_train, y_test = pca.get_mamm()
print(clusters_to_csv(kmeans.run(X, 7)))
tc, X, X_test, y_train, y_test = ica.get_mamm()
print(clusters_to_csv(kmeans.run(X, 2)))
tc, X, X_test, y_train, y_test = random_projection.get_mamm()
print(clusters_to_csv(kmeans.run(X, 2)))
tc, X, X_test, y_train, y_test = factor_an.get_mamm()
print(clusters_to_csv(kmeans.run(X, 7)))
tc, X, X_test, y_train, y_test = pca.get_mamm()
print(clusters_to_csv(exmax.run(X, 7)))
tc, X, X_test, y_train, y_test = ica.get_mamm()
print(clusters_to_csv(exmax.run(X, 6)))
tc, X, X_test, y_train, y_test = random_projection.get_mamm()
print(clusters_to_csv(exmax.run(X, 7)))
tc, X, X_test, y_train, y_test = factor_an.get_mamm()
print(clusters_to_csv(exmax.run(X, 7)))
'''


train_percentages = [20, 40, 60, 80]

print('kmeans pca nn mamm')
#cluster dimension reduced nn
for train_percentage in train_percentages:
    train_count, train_attributes, train_labels, test_attributes, test_labels = get_kmeans_pca(train_percentage)
    nn.run(train_attributes, train_labels, test_attributes, test_labels, train_percentage)

print('kmeans ica nn mamm')
#cluster dimension reduced nn
for train_percentage in train_percentages:
    train_count, train_attributes, train_labels, test_attributes, test_labels = get_kmeans_ica(train_percentage)
    nn.run(train_attributes, train_labels, test_attributes, test_labels, train_percentage)

print('kmeans rp nn mamm')
#cluster dimension reduced nn
for train_percentage in train_percentages:
    train_count, train_attributes, train_labels, test_attributes, test_labels = get_kmeans_rp(train_percentage)
    nn.run(train_attributes, train_labels, test_attributes, test_labels, train_percentage)

print('kmeans fa nn mamm')
#cluster dimension reduced nn
for train_percentage in train_percentages:
    train_count, train_attributes, train_labels, test_attributes, test_labels = get_kmeans_fa(train_percentage)
    nn.run(train_attributes, train_labels, test_attributes, test_labels, train_percentage)

print('em pca nn mamm')
#cluster dimension reduced nn
for train_percentage in train_percentages:
    train_count, train_attributes, train_labels, test_attributes, test_labels = get_exmax_pca(train_percentage)
    nn.run(train_attributes, train_labels, test_attributes, test_labels, train_percentage)

print('em ica nn mamm')
#cluster dimension reduced nn
for train_percentage in train_percentages:
    train_count, train_attributes, train_labels, test_attributes, test_labels = get_exmax_ica(train_percentage)
    nn.run(train_attributes, train_labels, test_attributes, test_labels, train_percentage)

print('em rp nn mamm')
#cluster dimension reduced nn
for train_percentage in train_percentages:
    train_count, train_attributes, train_labels, test_attributes, test_labels = get_exmax_rp(train_percentage)
    nn.run(train_attributes, train_labels, test_attributes, test_labels, train_percentage)

print('em fa nn mamm')
#cluster dimension reduced nn
for train_percentage in train_percentages:
    train_count, train_attributes, train_labels, test_attributes, test_labels = get_exmax_fa(train_percentage)
    nn.run(train_attributes, train_labels, test_attributes, test_labels, train_percentage)