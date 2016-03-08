function    [cluster_labels,ppca_label,mse,time_mppca,time_smmc,time_sc,W] = smmc(X,nClusts,ppca_dim,ncentres,knn,power)

%%%% 输入 注意，knn和ncentres这两个参数太大会导致矩阵条件数变差的。
%  X            N行D列矩阵
%  nClusts      聚类数 
%  ppca_dim     切空间维数
%  ncentres     MPPCA分析器个数
%  knn          每个点K近邻包含点的个数
%  power        主角度余弦值乘积之幂


%%%% 输出
%  cluster_labels      聚类标签    
%  ppca_label          分析器标签
%  mse                 均方误差
%  time_mppca          MPPCA消耗时间
%  time_smmc           SMMC消耗时间
%  time_sc             SC消耗时间
%  W                   相似性矩阵W
%   参数的含义参考我们的论文

[D,N] = size(X);
if nargin < 6
    power = 8;
end
if nargin < 5
    knn = 2*round(log(size(X,2)));
end
if nargin < 4
    ncentres = floor(N/(20*ppca_dim));
end

data = X';

t1  = clock;
disp('用混合概率主成分分析器生成混合模型中。。。');
mix = gmm(D, ncentres, 'ppca', ppca_dim);

options = foptions;
options(14) = 10; % knn只迭代十次
options(1) = -1;  
mix = gmminit(mix, data, options);

options(1)  = -1;		
options(14) = 50;		
[mix,options, errlog] = gmmem(mix, data, options);

a = gmmactiv(mix, data);
[uu,vv] = sort(a');
ppca_label = vv(end,:);
time_mppca = etime(clock,t1);

t2 = clock;
for k = 1:ncentres
    ctr{k,1} = mix.centres(k,:);
    dir{k,1} = mix.U(:,:,k)';
end;
mse = computing_L2_error(data, ppca_dim, ppca_label, ctr, dir);

fprintf(1,'  -->计算相似性矩阵.\n'); 
UU = zeros(ncentres,ncentres);
for i = 1:ncentres;
    for j = 1:ncentres;
%         UU(i,j) = prod(cos(subspacea(mix.U(:,:,i),mix.U(:,:,j))));
        UU(i,j) = prod(flipud(svd(mix.U(:,:,j)'*mix.U(:,:,i))));
    end;
end;

X2 = sum(X.^2,1);
Distance = repmat(X2,N,1)+repmat(X2',1,N)-2*X'*X;
[sorted,index] = sort(Distance);
G = zeros(N,N);
for ii = 1:N
   for jj = index(2:knn+1,ii);
       G(ii,jj) = UU(ppca_label(1,ii),ppca_label(1,jj)).^power;
   end;
end
W = max(G,G');

fprintf(1,'  -->计算前k个广义特征向量.\n'); 
DD = diag(sum(W,2));
OPTS.disp = 0;
N1 = min(20,size(W,1));
[eigvector, eigvalue] = eigs(DD-W,DD,N1,'sa',OPTS);
eigvalue = diag(eigvalue);
[junk, ind] = sort(eigvalue);
eigvalue = eigvalue(ind);
eigvector = eigvector(:, ind);
V = eigvector(:,1:nClusts);
    
fprintf(1,'  -->kmeans.\n');
cluster_labels = kmeans(V,nClusts,'EmptyAction','drop','Replicates',10);
time_sc = etime(clock,t2);
time_smmc = etime(clock,t1);







function mse = computing_L2_error(data, dim, idx, ctr, dir)

%  Modified by yong wang (yongwang82@gmail.com)

D = size(data,2);

K = max(idx);

if length(dim) == 1 && K > 1
    dim = dim*ones(K,1);
end


mse = 0;
for k = 1:K
    cls_k = data((idx==k),:);
    n_k = size(cls_k,1);
    if n_k > dim(k)
        mse = mse + sum(sum(((cls_k - repmat(ctr{k,1},n_k,1))*(eye(D) - dir{k,1}'*dir{k,1})).^2,2));
    end
end
function errstring = consist(model, type, inputs, outputs)

errstring = '';


if ~isempty(type)
  if ~isfield(model, 'type')
    errstring = 'Data structure does not contain type field';
    return
  end
  s = model.type;
  if ~strcmp(s, type)
    errstring = ['Model type ''', s, ''' does not match expected type ''',...
	type, ''''];
    return
  end
end

if nargin > 2
  if ~isfield(model, 'nin')
    errstring = 'Data structure does not contain nin field';
    return
  end

  data_nin = size(inputs, 2);
  if model.nin ~= data_nin
    errstring = ['Dimension of inputs ', num2str(data_nin), ...
	' does not match number of model inputs ', num2str(model.nin)];
    return
  end
end

if nargin > 3
  if ~isfield(model, 'nout')
    errstring = 'Data structure does not conatin nout field';
    return
  end
  data_nout = size(outputs, 2);
  if model.nout ~= data_nout
    errstring = ['Dimension of outputs ', num2str(data_nout), ...
	' does not match number of model outputs ', num2str(model.nout)];
    return
  end


  num_in = size(inputs, 1);
  num_out = size(outputs, 1);
  if num_in ~= num_out
    errstring = ['Number of input patterns ', num2str(num_in), ...
	' does not match number of output patterns ', num2str(num_out)];
    return
  end
end
function n2 = dist2(x, c)

[ndata, dimx] = size(x);
[ncentres, dimc] = size(c);
if dimx ~= dimc
	error('Data dimension does not match dimension of centres')
end

n2 = (ones(ncentres, 1) * sum((x.^2)', 1))' + ...
  ones(ndata, 1) * sum((c.^2)',1) - ...
  2.*(x*(c'));

% Rounding errors occasionally cause negative entries in n2
if any(any(n2<0))
  n2(n2<0) = 0;
end
function [evals, evec] = eigdec(x, N)
if nargout == 1
   evals_only = logical(1);
else
   evals_only = logical(0);
end

if N ~= round(N) | N < 1 | N > size(x, 2)
   error('Number of PCs must be integer, >0, < dim');
end

% Find the eigenvalues of the data covariance matrix
if evals_only
   % Use eig function as always more efficient than eigs here
   temp_evals = eig(x);
else
   % Use eig function unless fraction of eigenvalues required is tiny
   if (N/size(x, 2)) > 0.04
      [temp_evec, temp_evals] = eig(x);
   else
      options.disp = 0;
      [temp_evec, temp_evals] = eigs(x, N, 'LM', options);
   end
   temp_evals = diag(temp_evals);
end

% Eigenvalues nearly always returned in descending order, but just
% to make sure.....
[evals perm] = sort(-temp_evals);
evals = -evals(1:N);
if ~evals_only
   if evals == temp_evals(1:N)
      % Originals were in order
      evec = temp_evec(:, 1:N);
      return
   else
      % Need to reorder the eigenvectors
      for i=1:N
         evec(:,i) = temp_evec(:,perm(i));
      end
   end
end
function mix = gmm(dim, ncentres, covar_type, ppca_dim)
if ncentres < 1
  error('Number of centres must be greater than zero')
end

mix.type = 'gmm';
mix.nin = dim;
mix.ncentres = ncentres;

vartypes = {'spherical', 'diag', 'full', 'ppca'};

if sum(strcmp(covar_type, vartypes)) == 0
  error('Undefined covariance type')
else
  mix.covar_type = covar_type;
end

% Make default dimension of PPCA subspaces one.
if strcmp(covar_type, 'ppca')
  if nargin < 4
    ppca_dim = 1;
  end
  if ppca_dim > dim
    error('Dimension of PPCA subspaces must be less than data.')
  end
  mix.ppca_dim = ppca_dim;
end

% Initialise priors to be equal and summing to one
mix.priors = ones(1,mix.ncentres) ./ mix.ncentres;

% Initialise centres
mix.centres = randn(mix.ncentres, mix.nin);

% Initialise all the variances to unity
switch mix.covar_type

case 'spherical'
  mix.covars = ones(1, mix.ncentres);
  mix.nwts = mix.ncentres + mix.ncentres*mix.nin + mix.ncentres;
case 'diag'
  % Store diagonals of covariance matrices as rows in a matrix
  mix.covars =  ones(mix.ncentres, mix.nin);
  mix.nwts = mix.ncentres + mix.ncentres*mix.nin + ...
    mix.ncentres*mix.nin;
case 'full'
  % Store covariance matrices in a row vector of matrices
  mix.covars = repmat(eye(mix.nin), [1 1 mix.ncentres]);
  mix.nwts = mix.ncentres + mix.ncentres*mix.nin + ...
    mix.ncentres*mix.nin*mix.nin;
case 'ppca'
  % This is the off-subspace noise: make it smaller than
  % lambdas
  mix.covars = 0.1*ones(1, mix.ncentres);
  % Also set aside storage for principal components and
  % associated variances
  init_space = eye(mix.nin);
  init_space = init_space(:, 1:mix.ppca_dim);
  init_space(mix.ppca_dim+1:mix.nin, :) = ...
    ones(mix.nin - mix.ppca_dim, mix.ppca_dim);
  mix.U = repmat(init_space , [1 1 mix.ncentres]);
  mix.lambda = ones(mix.ncentres, mix.ppca_dim);
  % Take account of additional parameters
  mix.nwts = mix.ncentres + mix.ncentres*mix.nin + ...
    mix.ncentres + mix.ncentres*mix.ppca_dim + ...
    mix.ncentres*mix.nin*mix.ppca_dim;
otherwise
  error(['Unknown covariance type ', mix.covar_type]);               
end
function a = gmmactiv(mix, x)
errstring = consist(mix, 'gmm', x);
if ~isempty(errstring)
  error(errstring);
end

ndata = size(x, 1);
a = zeros(ndata, mix.ncentres); 

switch mix.covar_type
  
case 'spherical'
  n2 = dist2(x, mix.centres);
  
  wi2 = ones(ndata, 1) * (2 .* mix.covars);
  normal = (pi .* wi2) .^ (mix.nin/2);
  
  a = exp(-(n2./wi2))./ normal;
  
case 'diag'
  normal = (2*pi)^(mix.nin/2);
  s = prod(sqrt(mix.covars), 2);
  for j = 1:mix.ncentres
    diffs = x - (ones(ndata, 1) * mix.centres(j, :));
    a(:, j) = exp(-0.5*sum((diffs.*diffs)./(ones(ndata, 1) * ...
      mix.covars(j, :)), 2)) ./ (normal*s(j));
  end
  
case 'full'
  normal = (2*pi)^(mix.nin/2);
  for j = 1:mix.ncentres
    diffs = x - (ones(ndata, 1) * mix.centres(j, :));
    c = chol(mix.covars(:, :, j));
    temp = diffs/c;
    a(:, j) = exp(-0.5*sum(temp.*temp, 2))./(normal*prod(diag(c)));
  end
case 'ppca'
  log_normal = mix.nin*log(2*pi);
  d2 = zeros(ndata, mix.ncentres);
  logZ = zeros(1, mix.ncentres);
  for i = 1:mix.ncentres
    k = 1 - mix.covars(i)./mix.lambda(i, :);
    logZ(i) = log_normal + mix.nin*log(mix.covars(i)) - ...
      sum(log(1 - k));
    diffs = x - ones(ndata, 1)*mix.centres(i, :);
    proj = diffs*mix.U(:, :, i);
    d2(:,i) = (sum(diffs.*diffs, 2) - ...
      sum((proj.*(ones(ndata, 1)*k)).*proj, 2)) / ...
      mix.covars(i);
  end
  a = exp(-0.5*(d2 + ones(ndata, 1)*logZ));
otherwise
  error(['未知协方差错误 ', mix.covar_type]);
end
  function [mix,options, errlog,new_pr] = gmmem(mix, x, options)
errstring = consist(mix, 'gmm', x);
if ~isempty(errstring)
  error(errstring);
end

[ndata, xdim] = size(x);

% Sort out the options
if (options(14))
  niters = options(14);
else
  niters = 100;
end

display = options(1);
store = 0;
if (nargout > 2)
  store = 1;	
  errlog = zeros(1, niters);
end
test = 0;
if options(3) > 0.0
  test = 1;	
end

check_covars = 0;
if options(5) >= 1
  if display >= 0
    disp('check_covars is on');
  end
  check_covars = 1;
  MIN_COVAR = eps;	
  init_covars = mix.covars;
end
for n = 1:niters
  
  [post, act] = gmmpost(mix, x);
  
  if (display | store | test)
    prob = act*(mix.priors)';
    e = - sum(log(prob));
    if store
      errlog(n) = e;
    end
    if display > 0
      fprintf(1, 'Cycle %4d  Error %11.6f\n', n, e);
    end
    if test
      if (n > 1 & abs(e - eold) < options(3))
        options(8) = e;
        return;
      else
        eold = e;
      end
    end
  end
  
  ss = sum(post, 1);
  if any(ss==0)
    warning('0先验概率')  
    zero_columns = find(ss==0);
    post(:, zero_columns) = 1e-4;
  end
  new_pr = sum(post, 1);
  new_c = post' * x;
  
  mix.priors = new_pr ./ ndata;   
  
  mix.centres = new_c ./ (new_pr' * ones(1, mix.nin));
  
  switch mix.covar_type
  case 'spherical'
    n2 = dist2(x, mix.centres);
    for j = 1:mix.ncentres
      v(j) = (post(:,j)'*n2(:,j));
    end
    mix.covars = ((v./new_pr))./mix.nin;
    if check_covars
      for j = 1:mix.ncentres
        if mix.covars(j) < MIN_COVAR
          mix.covars(j) = init_covars(j);
        end
      end
    end
  case 'diag'
    for j = 1:mix.ncentres
      diffs = x - (ones(ndata, 1) * mix.centres(j,:));
      mix.covars(j,:) = sum((diffs.*diffs).*(post(:,j)*ones(1, ...
        mix.nin)), 1)./new_pr(j);
    end
    if check_covars
      for j = 1:mix.ncentres
        if min(mix.covars(j,:)) < MIN_COVAR
          mix.covars(j,:) = init_covars(j,:);
        end
      end
    end
  case 'full'
    for j = 1:mix.ncentres
      diffs = x - (ones(ndata, 1) * mix.centres(j,:));
      diffs = diffs.*(sqrt(post(:,j))*ones(1, mix.nin));
      mix.covars(:,:,j) = (diffs'*diffs)/new_pr(j);
    end
    if check_covars
      for j = 1:mix.ncentres
        if min(svd(mix.covars(:,:,j))) < MIN_COVAR
          mix.covars(:,:,j) = init_covars(:,:,j);
        end
      end
    end
  case 'ppca'
    for j = 1:mix.ncentres
      diffs = x - (ones(ndata, 1) * mix.centres(j,:));
      diffs = diffs.*(sqrt(post(:,j))*ones(1, mix.nin));
      [tempcovars, tempU, templambda] = ...
	ppca((diffs'*diffs)/new_pr(j), mix.ppca_dim);
      if length(templambda) ~= mix.ppca_dim
	error('成分不足');
      else 
        mix.covars(j) = tempcovars;
        mix.U(:, :, j) = tempU;
        mix.lambda(j, :) = templambda;
      end
    end
    if check_covars
      if mix.covars(j) < MIN_COVAR
        mix.covars(j) = init_covars(j);
      end
    end
    otherwise
      error(['未知协方差错误 ', mix.covar_type]);               
  end
end

options(8) = -sum(log(gmmprob(mix, x)));
if (display >= 0)
  disp(maxitmess);
end
  function mix = gmminit(mix, x, options)
[ndata, xdim] = size(x);
errstring = consist(mix, 'gmm', x);
if ~isempty(errstring)
  error(errstring);
end
GMM_WIDTH = 1.0;

options(5) = 1;	
[mix.centres, options, post] = kmeans1(mix.centres, x, options);

l = 1;
while min(sum(post, 1)) < mix.ppca_dim+1 && l < 10;
    fprintf(1,'-->The smallest cluster has %d points, reuse kmeans algorithm to set centres\n',min(sum(post, 1)));
    options(5) = 1;	
    [mix.centres, options, post] = kmeans1(mix.centres, x, options);
    l = l+1;
end;
cluster_sizes = max(sum(post, 1), 1);  
mix.priors = cluster_sizes/sum(cluster_sizes); 

switch mix.covar_type
case 'spherical'
   if mix.ncentres > 1
      cdist = dist2(mix.centres, mix.centres);
      cdist = cdist + diag(ones(mix.ncentres, 1)*realmax);
      mix.covars = min(cdist);
      mix.covars = mix.covars + GMM_WIDTH*(mix.covars < eps);
   else
      mix.covars = mean(diag(cov(x)));
   end
  case 'diag'
    for j = 1:mix.ncentres
      c = x(find(post(:, j)),:);
      diffs = c - (ones(size(c, 1), 1) * mix.centres(j, :));
      mix.covars(j, :) = sum((diffs.*diffs), 1)/size(c, 1);
      mix.covars(j, :) = mix.covars(j, :) + GMM_WIDTH.*(mix.covars(j, :)<eps);
    end
  case 'full'
    for j = 1:mix.ncentres
      c = x(find(post(:, j)),:);
      diffs = c - (ones(size(c, 1), 1) * mix.centres(j, :));
      mix.covars(:,:,j) = (diffs'*diffs)/(size(c, 1));
      if rank(mix.covars(:,:,j)) < mix.nin
	mix.covars(:,:,j) = mix.covars(:,:,j) + GMM_WIDTH.*eye(mix.nin);
      end
    end
  case 'ppca'
    for j = 1:mix.ncentres
      c = x(find(post(:,j)),:);
      diffs = c - (ones(size(c, 1), 1) * mix.centres(j, :));
      [tempcovars, tempU, templambda] = ...
	ppca((diffs'*diffs)/size(c, 1), mix.ppca_dim);
      if length(templambda) ~= mix.ppca_dim
	    error('成分不足');
      else 
        mix.covars(j) = tempcovars;
        mix.U(:, :, j) = tempU;
        mix.lambda(j, :) = templambda;
      end
    end
  otherwise
    error(['未知协方差错误 ', mix.covar_type]);
end
function [post, a] = gmmpost(mix, x)
errstring = consist(mix, 'gmm', x);
if ~isempty(errstring)
  error(errstring);
end

ndata = size(x, 1);

a = gmmactiv(mix, x);

post = (ones(ndata, 1)*mix.priors).*a;
s = sum(post, 2);
if any(s==0)
   warning('Some zero posterior probabilities')
   % Set any zeros to one before dividing
   zero_rows = find(s==0);
   s = s + (s==0);
   post(zero_rows, :) = 1/mix.ncentres;
end
post = post./(s*ones(1, mix.ncentres));

function prob = gmmprob(mix, x)
errstring = consist(mix, 'gmm', x);
if ~isempty(errstring)
  error(errstring);
end
a = gmmactiv(mix, x);
prob = a * (mix.priors)';
function [centres, options, post, errlog] = kmeans1(centres, data, options)
[ndata, data_dim] = size(data);
[ncentres, dim] = size(centres);
if dim ~= data_dim
  error('维数错误')
end

if (ncentres > ndata)
  error('knn中心点过多')
end

% Sort out the options
if (options(14))
  niters = options(14);
else
  niters = 100;
end

store = 0;
if (nargout > 3)
  store = 1;
  errlog = zeros(1, niters);
end


if (options(5) == 1)
   perm = randperm(ndata);
   perm = perm(1:ncentres);
   centres = data(perm, :);
end

id = eye(ncentres);

for n = 1:niters

  old_centres = centres;
  d2 = dist2(data, centres);
  [minvals, index] = min(d2', [], 1);
  post = id(index,:);

  num_points = sum(post, 1);
  for j = 1:ncentres
    if (num_points(j) > 0)
      centres(j,:) = sum(data(find(post(:,j)),:), 1)/num_points(j);
    end
  end

  e = sum(minvals);
  if store
    errlog(n) = e;
  end
  if options(1) > 0
    fprintf(1, 'Cycle %4d  Error %11.6f\n', n, e);
  end

  if n > 1
    if max(max(abs(centres - old_centres))) < options(2) & ...
        abs(old_e - e) < options(3)
      options(8) = e;
      return;
    end
  end
  old_e = e;
end

options(8) = e;
if (options(1) >= 0)
  disp(maxitmess);
end
function s = maxitmess()
s = '超出最大迭代数';
function [var, U, lambda] = ppca(x, ppca_dim)
if ppca_dim ~= round(ppca_dim) | ppca_dim < 1 | ppca_dim > size(x, 2)
   error('PC数非整, >0, < dim');
end

[ndata, data_dim] = size(x);
% Assumes that x is centred and responsibility weighted
% covariance matrix
[l Utemp] = eigdec(x, data_dim);
% Zero any negative eigenvalues (caused by rounding)
l(l<0) = 0;
% Now compute the sigma squared values for all possible values
% of q
s2_temp = cumsum(l(end:-1:1))./[1:data_dim]';
% If necessary, reduce the value of q so that var is at least
% eps * largest eigenvalue
q_temp = min([ppca_dim; data_dim-min(find(s2_temp/l(1) > eps))]);
if q_temp ~= ppca_dim
  wstringpart = '协方差矩阵条件数差，要不重新设一下参数？';
  wstring = sprintf('%s %d/%d PCs', ...
      wstringpart, q_temp, ppca_dim);
  warning(wstring);
end

% make sure no 0/0 or something like 1/0 (i.e., NaN or Inf) appear in
% gmmactiv.m. otherwise, eigdec.m will not work.

% make sure var>0, and each value in lambda is larger than var;
if q_temp < ppca_dim;
  nz = min(find (l < 1e-2));
  if nz > ppca_dim
    l(nz:end) = 1e-2;
  else 
    l(nz:ppca_dim) = 1e-1;
    l(ppca_dim+1:end) = 1e-2;
  end;
end;
var = mean(l(ppca_dim+1:end)); 
U = Utemp(:, 1:ppca_dim);
lambda(1:ppca_dim) = l(1:ppca_dim);

