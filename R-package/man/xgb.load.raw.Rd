% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/xgb.load.raw.R
\name{xgb.load.raw}
\alias{xgb.load.raw}
\title{Load serialised XGBoost model from R's raw vector}
\usage{
xgb.load.raw(buffer)
}
\arguments{
\item{buffer}{The buffer returned by \code{\link[=xgb.save.raw]{xgb.save.raw()}}.}
}
\description{
User can generate raw memory buffer by calling \code{\link[=xgb.save.raw]{xgb.save.raw()}}.
}
