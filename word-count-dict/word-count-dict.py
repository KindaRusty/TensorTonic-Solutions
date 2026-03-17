def word_count_dict(sentences):
    """
    Returns: dict[str, int] - global word frequency across all sentences
    """
    my_dict = {}
    
    #1: Lặp từng câu
    for sentence in sentences:
        #2: lặp từng từ
        for word in sentence:
            #3: kiểm tra 
            if word in my_dict:
                my_dict[word] +=1 #ko dùng string vì nó sẽ print ra chính đó luôn 
            else:
                my_dict[word] = 1
    return my_dict