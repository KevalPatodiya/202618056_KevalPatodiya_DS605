import scrapy


class MySpider(scrapy.Spider):
    name = "books"

    allowed_domains = ["books.toscrape.com"]

    start_urls = [
        "https://books.toscrape.com/catalogue/page-1.html"
    ]

    page_count = 1
    max_pages = 5        # Scrape first 5 pages (100 books)

    def parse(self, response):

        # Visit each book page
        for book in response.css("article.product_pod"):
            book_url = response.urljoin(book.css("h3 a::attr(href)").get())
            yield scrapy.Request(book_url, callback=self.parse_book)

        # Follow pagination until page 5
        if self.page_count < self.max_pages:
            next_page = response.css("li.next a::attr(href)").get()

            if next_page:
                self.page_count += 1
                yield response.follow(next_page, callback=self.parse)

    def parse_book(self, response):

        # Extract values from product information table
        table = {}

        for row in response.css("table.table-striped tr"):
            key = row.css("th::text").get()
            value = row.css("td::text").get()
            table[key] = value

        yield {
            "title": response.css("div.product_main h1::text").get(),
            "category": response.css(
                "ul.breadcrumb li:nth-child(3) a::text"
            ).get(),
            "price": table.get("Price (incl. tax)"),
            "rating": response.css(
                "p.star-rating::attr(class)"
            ).get().replace("star-rating ", ""),
            "availability": table.get("Availability"),
            "product_description": response.css(
                "#product_description + p::text"
            ).get(),
            "UPC": table.get("UPC"),
            "number_of_reviews": table.get("Number of reviews"),
            "product_url": response.url,
        }