# The Python scheduler writes job.json in this run's explicit output folder.
# No truth, environment-variable path, or command-line parameter is read here.
library(jsonlite)
job <- fromJSON('job.json')
.libPaths(c(job$library, .libPaths()))
set.seed(job$seed)
x <- as.matrix(read.csv(job$expression, row.names=1, check.names=FALSE))
tfs <- read.csv(job$tfs, colClasses='character')$tf
targets <- setdiff(colnames(x),tfs)
stopifnot(all(is.finite(x)),length(tfs)>1,length(targets)>0)
original_genes <- colnames(x)
variable <- apply(x,2,sd)>0
# Constant genes cannot be estimated; preserve their candidate edges with score=0.
x <- x[,variable,drop=FALSE]
tf_used <- intersect(tfs,colnames(x)); target_used <- intersect(targets,colnames(x))
stopifnot(length(tf_used)>1,length(target_used)>0)
params <- job$params
if (job$method=='TIGRESS') {
  library(lars)
  source(file.path(job$software,'stabilityselection.R'))
  source(file.path(job$software,'tigress.R'))
  score <- tigress(x,tflist=tf_used,targetlist=target_used,alpha=params$alpha,
                   nstepsLARS=params$nstepsLARS,nsplit=params$nsplit,
                   scoring='area',allsteps=FALSE,usemulticore=FALSE)
} else if (job$method=='CLR') {
  score <- minet::minet(x,method='clr',estimator=params$estimator,
                       disc=params$disc,nbins=params$nbins)[tf_used,target_used,drop=FALSE]
} else if (job$method=='PPCOR') {
  score <- abs(ppcor::pcor(x,method=params$method)$estimate)[tf_used,target_used,drop=FALSE]
} else stop('Unregistered official R method')
stopifnot(all(is.finite(score)))
full <- matrix(0,length(tfs),length(targets),dimnames=list(tfs,targets))
full[tf_used,target_used] <- score
out <- expand.grid(tf=tfs,target=targets,stringsAsFactors=FALSE)
out$score <- as.vector(full)
write.csv(out,'official_grn.csv',row.names=FALSE)
write_json(list(method=job$method,constant_genes=original_genes[!variable],
                projection='TF-to-nonTF candidate projection; CLR/PPCOR are symmetric association methods',
                R=R.version.string),'official_audit.json',auto_unbox=TRUE,pretty=TRUE)
