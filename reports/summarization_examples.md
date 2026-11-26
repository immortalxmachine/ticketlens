# Summarization examples (DialogSum test, first 5 of the fixed subset)

## test_372_1

**Dialogue**

```text
#Person1#: Jack, I think it's best that I actively ask for the results of interview.
#Person2#: That's a good idea! So you don't need to worry about it.
#Person1#: I think so, too. What do you think about that I write a letter of inquiry to the company?
#Person2#: It's OK. You could send an email to the employer via the internet.
#Person1#: Good idea. It is convenient and fast.
#Person2#: You could inquire about the results directly, but should be politely.
#Person1#: Certainly.
#Person2#: I think the letter of inquiry should be written briefly. If it's too long, I'm afraid the employer won't have time to read it.
#Person1#: I think that about one hundred words are enough.
#Person2#: After you send the letter of inquiry, you should pay close attention not to miss any reply.
#Person1#: OK, I know now.
```

| Source | Summary |
|---|---|
| **Human reference** | #Person1# wants to know the result of a interview. Jack suggests writing a polite and brief email to the company. |
| bart | #Person1#: Jack, I think it's best that I actively ask for the results of interview. #Person2#: That's a good idea! So you don't need to worry about it. |
| flan-t5 | #Person1#: I think it's best to ask for the results of the interview. |
| lead-2 | #Person1#: Jack, I think it's best that I actively ask for the results of interview. #Person2#: That's a good idea! So you don't need to worry about it. |

## test_456_1

**Dialogue**

```text
#Person1#: Wow, there are so many lanterns to appreciate. Now, I can see why it's called the Lantern Festival. It deserves its name.
#Person2#: Yeah. People always enjoy the lighted lanterns and the gala performances.
#Person1#: What are they doing over there? People keep on gathering there.
#Person2#: Did you notice the characters on the lanterns?
#Person1#: Sure. But you know that I can't read any Chinese characters. What do they say?
#Person2#: They are puzzles. It's a tradition to solve the puzzles on the lanterns during the Lantern Festival.
#Person1#: Very interesting. But I'm afraid we'd better do something else. Hey, look! There is a huge lantern there. Let's get close to it.
#Person2#: It's really eye-catching. It's the biggest dragon lantern I've ever seen in all my life.
#Person1#: Really? Then I'm really lucky. Oh, it's spewing fireworks from its huge mouth.
#Person2#: Very impressive. It's made of glass which makes it even brighter.
#Person1#: There are many Chinese characters on its body, too. What are they about? Puzzles?
#Person2#: Let me have a look. Oh, no. They are Chinese poems which describe this happy scene.
```

| Source | Summary |
|---|---|
| **Human reference** | #Person1# and #Person2# are appreciating lanterns. #Person2# explains the customs of the Lantern Festival and Chinese characters on the lanterns to #Person1#. |
| bart | The Lantern Festival is held every year in China. The festival is known for its lighted lanterns and gala performances. |
| flan-t5 | People gather at the Lantern Festival to enjoy the lighted lanterns. |
| lead-2 | #Person1#: Wow, there are so many lanterns to appreciate. Now, I can see why it's called the Lantern Festival. It deserves its name. #Person2#: Yeah. People always enjoy the lighted lanterns and the gala performances. |

## test_140_3

**Dialogue**

```text
#Person1#: Honey, of course I forgive you! I love you so much! I've really missed you. I was wrong to get upset over nothing. 
#Person2#: I'm sorry I haven't called or anything, but right after you decided you wanted a break, I was called up north to put out some major forest fires! I was in the middle of nowhere, working day and night, trying to prevent the blaze from spreading! It was pretty intense. 
#Person1#: Oh, honey, I'm glad you're okay! But I have some exciting news. . . I think I'm pregnant! 
#Person2#: Really? Wow, that's amazing! This is great news! I've always wanted to be a father! We'll go to the doctor first thing in the morning! 
#Person3#: We have your test results back and, indeed, you are pregnant. Let's see here. . . everything seems to be in order. Your approximate due date is October twenty-seventh two thousand and nine, so that means that the baby was conceived on February third, two thousand and nine. 
#Person2#: Are you sure? Are these things accurate? 
#Person3#: Well, yes sir, they are. 
#Person1#: What's wrong? Why are you asking these questions? 
#Person2#: This baby isn't mine! I was away the first week of February at a training seminar! 
#Person1#: I. . . I. . . no, it can't be. . . 
```

| Source | Summary |
|---|---|
| **Human reference** | #Person1# apologises to #Person2# after the quarrel and tells #Person2# she's pregnant. #Person2# feels so happy and they decide to see the doctor. After the pregnancy test about the date, #Person2# finds that it is not his baby, because he was away the first week of February at a training seminar. |
| bart | #Person1#: Honey, of course I forgive you! I love you so much! I've really missed you. #Person2#: I'm sorry I haven't called or anything, but I was called up north to put out some major forest fires! #Person3#: We have your test results back and, indeed, you are pregnant. |
| flan-t5 | The baby was conceived on February 3rd, two thousand and nine. |
| lead-2 | #Person1#: Honey, of course I forgive you! I love you so much! I've really missed you. I was wrong to get upset over nothing. #Person2#: I'm sorry I haven't called or anything, but right after you decided you wanted a break, I was called up north to put out some major forest fires! I was in the middle of nowhere, working day and night, trying to prevent the blaze from spreading! It was pretty intense. |

## test_137_3

**Dialogue**

```text
#Person1#: I wonder if you could do me a favor. 
#Person2#: I'd be happy to help you. Name it. 
#Person1#: My printer is broken. Can you print up a copy for me? 
#Person2#: Sure, no problem. Do you want me to edit it for you? 
#Person1#: Please, you write much better than I do. 
#Person2#: Oh, its nothing. 
```

| Source | Summary |
|---|---|
| **Human reference** | #Person1# asks for #Person2#'s help to print up a copy. #Person2# accepts and edits it for #Person1#. |
| bart | #Person1#: My printer is broken. Can you print up a copy for me? #Person2#: Sure, no problem. Do you want me to edit it for you? |
| flan-t5 | #Person1 wants to print a copy of the book, but the printer is broken. Then Person2 will edit it for him. |
| lead-2 | #Person1#: I wonder if you could do me a favor. #Person2#: I'd be happy to help you. Name it. |

## test_150_2

**Dialogue**

```text
#Person1#: What's up?
#Person2#: I guess there is some kind of virus seeking into my computer, I can't send out this e-mail. Do you have the number of the text port?
#Person1#: Do you mind I have a look at your computer?
#Person2#: Of course not, I appreciate that.
#Person1#: Well, it has nothing to do with virus. The problem is your attachment is a bit larger. It has exceeded the e-mail capacity.
#Person2#: I see. What can I do now?
#Person1#: You can send a compressed one.
```

| Source | Summary |
|---|---|
| **Human reference** | #Person2#'s attachment exceeds the e-mail capacity, #Person1# suggests compressing it. |
| bart | #Person1#: What's up?#Person2#: I guess there is some kind of virus seeking into my computer, I can't send out this e-mail. Do you have the number of the text port? |
| flan-t5 | #Person1#: The problem is that the attachment is too large. |
| lead-2 | #Person1#: What's up? #Person2#: I guess there is some kind of virus seeking into my computer, I can't send out this e-mail. Do you have the number of the text port? |
