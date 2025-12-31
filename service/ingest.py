from service.main import DocumentIndexer, configure_logging


def main() -> None:
    configure_logging()
    indexer = DocumentIndexer()
    indexer.scan_once()


if __name__ == "__main__":
    main()
