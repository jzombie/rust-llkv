use core::fmt;

#[derive(Debug, PartialEq, Eq)]
pub enum Error {
    Corrupt(&'static str),
    MissingPage,
    Pager(&'static str),
    Invalid(&'static str),
    KeyNotFound,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Corrupt(msg) => write!(f, "corrupt: {msg}"),
            Error::MissingPage => write!(f, "missing page"),
            Error::Pager(msg) => write!(f, "pager error: {msg}"),
            Error::Invalid(msg) => write!(f, "invalid: {msg}"),
            Error::KeyNotFound => write!(f, "key not found"),
        }
    }
}

impl std::error::Error for Error {}

impl From<std::io::Error> for Error {
    fn from(_: std::io::Error) -> Self {
        Error::Pager("io")
    }
}
