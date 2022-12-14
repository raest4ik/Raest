o
    NiDc	  ?                   @   s?   d dl Z d dlZd dlZd dlmZ d dlZejdd?adede	fdd?Z
d	d
? adefdd?adefdd?ZedkrBee? ? dS dS )?    N)?Counter?ru)?lang?s?lc                 c   sR   ? t ?| |?D ]}t|?D ]}d?|?d |? }|?? s"t?|?s"q|V  qqd S )N? )?	itertools?combinations_with_replacement?range?join?isdigit?morphyZword_is_known)r   r   ?i?j?out? r   ?5   c:\Users\Никита\Desktop\programm\get_filters.py?	get_combs   s   ?
??r   c                 C   s>   d}| D ]}d}|D ]}||vrd}q
|r||7 }q |S |S )Nr   TFr   )?str1?strs?resr   ?a?s1r   r   r   ?get_crossing   s   ?
r   c                 C   s0   t ?| ?d }tttdd? |j???}t| |?S )Nr   c                 S   s   | j S )N)?word)?xr   r   r   ?<lambda>   s    zword_process.<locals>.<lambda>)r   ?parse?list?set?mapZlexemer   )r   r   r   r   r   r   ?word_process   s   
r!   ?pathc                 C   s?  g }g }d}g }g }t | ddd???}tj|ddd?}t|?dd ? }t|?D ]\}	}
t?d	d
t?dd
|
d ???? }|?|? q&t	t
d
?|??? ?d
???}|?? D ]\}}|dkret|?dkre|?|? qRt|?D ]\}}|dkrt q?t|?}t|?dkr?|?|? qjt	t
tt?d	dd?|???? ???}|?? D ]\}}|dkr?|dk r?|?|? q?t|?D ]\}}|dkr? q?|?|? q?d
?|??? }tdd?D ]}	|?|	?}|dkr?|?|	? q?t|?W  d   ? S 1 s?w   Y  d S )Nr   ?rzutf-8)?encoding?,?")?	delimiter?	quotechar?   z\s+? u   [^a-zA-Zа-яёА-ЯЁ\d]?   ?   i?  ?2   i'  i?  ?
0123456789?   )?open?csv?readerr   ?	enumerate?re?sub?strip?append?dictr   r   ?lower?split?items?lenr!   r   ?count?sorted)r"   ?filtersZwords_lists?text?words?symbols?fr2   ?linesr   ?row?line?key?value?n?wr#   Ztext_2r   ?t?br   r   r   ?get_filters#   sP    
?
?$
?

?$?rM   ?__main__)r1   r4   r   ?collectionsr   Z	pymorphy2ZMorphAnalyzerr   ?str?intr   r   r!   rM   ?__name__?printr   r   r   r   ?<module>   s    	*?