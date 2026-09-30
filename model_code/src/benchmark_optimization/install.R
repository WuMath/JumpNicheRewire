# Called by preprocess_jumpniche.py; no user command-line parameters.
library(jsonlite)
settings <- fromJSON('install_settings.json')
.libPaths(c(settings$library, .libPaths()))
options(timeout=300)
for (p in c('lars','ppcor','infotheo')) {
  if (!requireNamespace(p, quietly=TRUE))
    install.packages(p, lib=settings$library, repos='https://cloud.r-project.org', type='binary')
}
if (!requireNamespace('minet', quietly=TRUE)) {
  # R 4.4 compatible official Bioconductor binary; no global environment updates.
  # Version verified from the official Bioconductor 3.20 PACKAGES.gz index.
  url <- 'https://bioconductor.org/packages/3.20/bioc/bin/windows/contrib/4.4/minet_3.64.0.zip'
  download.file(url, 'minet_3.64.0.zip', mode='wb')
  install.packages('minet_3.64.0.zip', repos=NULL, type='win.binary', lib=settings$library)
  write_json(list(url=url,md5=unname(tools::md5sum('minet_3.64.0.zip'))),
             'minet_download.json',auto_unbox=TRUE,pretty=TRUE)
}
packages <- c('lars','ppcor','minet','infotheo','jsonlite')
stopifnot(all(vapply(packages, requireNamespace, logical(1), quietly=TRUE)))
versions <- setNames(lapply(packages,function(p) as.character(packageVersion(p))),packages)
versions$R <- R.version.string
write_json(versions,'installed_versions.json',auto_unbox=TRUE,pretty=TRUE)
