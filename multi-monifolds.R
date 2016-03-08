library(MetabolAnalyze)

library(R.matlab)
x<-readMat("EX2/2d.mat")$data
N<-ncol(x)

M<-round(N/10)
d<-1
Dim<-2
K<-round(2*log(N))
k<-2
O<-8

cppcafit<-mppca.metabol(t(x),d,d,M,M)
plot(t(cppcafit$mean))
v_vec<-matrix(cppcafit$loadings,ncol=2)
plot(v_vec)

#normal_vec<-apply(v_vec,1,function(n){rowSums(solve(rbind(n,rep(1,Dim))))})
normal_vec<-t(v_vec)
normal_denum<-diag(1/sqrt(colSums(normal_vec^2)))
normal_vec<-t(normal_vec%*%normal_denum)
normal_vec<-abs(normal_vec%*%t(normal_vec))^O
distance<-as.matrix(dist(t(x),upper=F))
knear_index<-apply(distance,2,function(ds) match(c(1:(K+1)),rank(ds)))
is_in<-matrix(rep(0,N*N),ncol=N)
for(i in 1:N) is_in[i,knear_index[,i]]<-1
w<-matrix(0,ncol=N,nrow=N)
for(i in 1:N)
  for(j in 1:N)
  {
    if(i!=j&&(is_in[i,j]==1||is_in[j,i]==1)) w[i,j]<-normal_vec[cppcafit$clustering[i],cppcafit$clustering[j]];
  }
E<-diag(rowSums(w))
L<-E-w
eigen_L<-eigen(L)
U<-eigen_L$vectors[,(N-k+1):N]
#U<-eigen_L$vectors[,1:k]
kmn<-kmeans(x=U,k)
plot(U)
plot(t(x[,kmn$cluster==2]),ylim=c(-2,2))
points(t(x[,kmn$cluster==1]),col="red")
#library(kernlab)

