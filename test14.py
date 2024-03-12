from docarray import BaseDoc, DocList
from docarray.typing import ImageUrl


class BannerDoc(BaseDoc):
    image: ImageUrl
    title: str
    description: str


class PageDoc(BaseDoc):
    banner: BannerDoc
    content: str


page1 = PageDoc(
    banner=BannerDoc(
        image='https://example.com/image1.png',
        title='Hello World',
        description='This is a banner',
    ),
    content='Hello world is the most used example in programming, but do you know that? ...',
)

page2 = PageDoc(
    banner=BannerDoc(
        image='https://example.com/image2.png',
        title='Bye Bye World',
        description='This is (distopic) banner',
    ),
    content='What if the most used example in programming was Bye Bye World, would programming be that much fun? ...',
)

docs = DocList[PageDoc]([page1, page2])

docs.summary()

