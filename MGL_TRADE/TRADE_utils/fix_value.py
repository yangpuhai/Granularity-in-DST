def fix_time(time):
    time0=[]
    if time != '':
        if time[0] == '0':
            time1 = time[1:]
            time0.append(time1)
        time0.append(time.replace(":",''))
        time0.append(time.replace(":", '.'))
        if ':' in time:
            hr = time.split(':')[0]
            mi = time.split(':')[1]
            if mi == '00':
                if ' ' in hr:
                    hr_int = int(hr.split(' ')[-1])
                else:
                    hr_int=int(hr)
                if hr_int > 12:
                    hr_2 = str(hr_int - 12)
                    time0.append(hr_2 + ' o\'clock')
                    time0.append(hr_2 + ' p.m')
                    time0.append(hr_2 + ' p.m.')
                    time0.append(hr_2 + ' pm')
                    time0.append(hr_2 + 'pm')
                    time0.append(hr_2 + ':00')
                    time0.append('0' + hr_2 + ':00')
                time0.append(hr)
            else:
                if ' ' in hr:
                    hr_int = int(hr.split(' ')[-1])
                else:
                    hr_int=int(hr)
                if hr_int > 12:
                    hr_3 = str(hr_int - 12)
                    time0.append(hr_3 + ':' + mi)
        else:
            if 'am' in time or 'pm' in time:
                if 'am' in time:
                    hr_4=time.replace('am','')
                    time0.append(hr_4 + ' am')
                if 'pm' in time:
                    hr_4=time.replace('pm','')
                    time0.append(hr_4 + ' pm')
            if len(time)==4:
                time0.append(time[:2] + ':' + time[2:])
        if time == '12:00':
            time0.append('around noon')
    return time0

fix_value_dict = {'guest house':['guesthouse','guestroom','gueshouse'],
                  'cheap':['inexpensive','less expensive','lower','budget'],
                  'moderate':['reasonable','medium','reasonably','not too high','middle','mid','not too expensive'],
                  'expensive':['pricey','upscale','luxurious'],
                  'christ college':['christ s college','christ s'],
                  'cafe uno':['caffe uno'],
                  'el shaddia guesthouse':['el shaddai'],
                  'centre':['center','centrally'],
                  'peterborough':['peterbourgh'],
                  'the man on the moon':['man on the moon'],
                  'london liverpool street':['london liverpool','liverpool street'],
                  'swimming pool':['water slide','swimmingpool','swimming','swim'],
                  'multiple sports':['sports'],
                  'kings college':['king s college'],
                  'saint johns college':['saint john s college'],
                  'kettles yard':['kettle s yard'],
                  'churchills college':['churchill college','churchhill college'],
                  'portugese':['portuguese','portugeuese'],
                  'a and b guest house':['a and b guesthouse','a and b quest house'],
                  'midsummer house restaurant':['midsummer house'],
                  'finches bed and breakfast':['finches bed & breakfast'],
                  'rosas bed and breakfast':['rosa s bed and breakfast','rosa s'],
                  'pizza hut city centre':['pizza hut city','pizza hut'],
                  'emmanuel college':['emmauel college'],
                  'london kings cross':['london king s cross','kings cross station in london','kings cross','london'],
                  'all saints church':['all saint s church'],
                  'pizza hut fenditton':['pizza hut fen ditton','pizza express fen ditton','fen ditton','pizza express'],
                  '09:00':['9am','9 am'],
                  'north american':['american'],
                  'lovell lodge':['lovell lodege'],
                  'sheeps green and lammas land park fen causeway':['sheep s green and lammas land park fen causeway'],
                  'milton country park':['milton county park'],
                  'express by holiday inn cambridge':['express by holiday inn'],
                  'warkworth house':['warworth house'],
                  'acorn guest house':['acorn','arcon guesthouse'],
                  'cambridge belfry':['caridge belfrey','cambridge belfrey','cambrdige belfry'],
                  'parkside pools':['parkside','swimmingpool'],
                  'theatre':['theater'],
                  'dojo noodle bar':['dojo noddle bar'],
                  'holy trinity church':['holy trinity'],
                  'huntingdon marriott hotel':['huntington marriott hotel','huntingdon marriot hotel','huntingdon marriott','hungtingdon marriott hotel','huntingdon marriot'],
                  'alexander bed and breakfast':['alexander bed and breafast','alexander'],
                  'king hedges learner pool':['kings hedges'],
                  'hobsons house':['hobson s house','hobson house'],
                  'norwich':['norwhich'],
                  '0':['zero','not rated','unrated'],
                  'australian':['australasian'],
                  'birmingham new street':['birmignham new street','birmingham new st','birmingham'],
                  'stazione restaurant and coffee bar':['stazione'],
                  'saturday':['staurday'],
                  'stevenage':['stevanage','steveanage'],
                  'cambridge and county folk museum':['cambridge country folk museum','county folk museum'],
                  'city stop restaurant':['citi stop restaurant'],
                  'cambridge contemporary art museum':['contemporary art'],
                  'nandos city centre':['nandos'],
                  'bangkok city':['bankok city'],
                  'peoples portraits exhibition at girton college':['people s portraits exhibition at girton college'],
                  'moderate|cheap':['mid to low range'],
                  'concert hall':['concerthall'],
                  'cambridge train station':['train station','cambridge'],
                  'arbury lodge guesthouse':['arbury lodge','arbury guesthouse'],
                  'limehouse':['lime house'],
                  'fitzbillies restaurant':['fitzbillies'],
                  'great saint marys church':['great saint mary s church'],
                  'thursday':['thusday','trusday','tursday'],
                  'stansted airport':['standsted airport'],
                  'university arms hotel':['university arms'],
                  'scudamores punting co':['scudmores punting co'],
                  'queens college':['queen s college'],
                  'museum of archaelogy and anthropology':['museum of archaeology and anthropology','museum of classical archaeology'],
                  'architecture':['architectural','architect'],
                  'gastropub':['astropub'],
                  'carolina bed and breakfast':['caroline bed and breakfast','carolina bed & breakfast'],
                  'frankie and bennys':['frankie and benny s'],
                  'ashley hotel':['ashely hotel','ashley'],
                  'queens':['queen s'],
                  'restaurant 2 two':['2 two'],
                  'little saint marys church':['little saint mary s church'],
                  'cambridge botanic gardens':['botanic gardens'],
                  'sheeps green and lammas land park':['sheep s green and lammas land park'],
                  'polynesian':['polunesian'],
                  'lan hong house':['ian hong house'],
                  'the junction':['junction'],
                  'wednesday':['wednsday','wedensday'],
                  'gonville hotel':['gonville'],
                  'peterborough train station':['peterborough'],
                  'cambridge':['cambidge'],
                  'kings lynn':['king s lynn','kings lane'],
                  'bloomsbury restaurant':['bloomsbury'],
                  'funky fun house':['funky funhouse'],
                  'riverside brasserie':['riverside brassiere'],
                  'cineworld cinema':['cineworld'],
                  'museum':['musems'],
                  'da vinci pizzeria':['da vinci pizzria'],
                  'anatolia':['antolia'],
                  'curry prince':['curry price'],
                  'sheeps green':['sheep s green'],
                  'caribbean':['carribean'],
                  'archway house':['artchway house'],
                  'pizza hut cherry hinton':['pizza hut cherry hilton'],
                  'italian':['italain'],
                  'the peking restaurant:':['the peking restaurant'],
                  'middle eastern':['muslim'],
                  'broxbourne':['broxburne','boxbourne'],
                  'nightclub':['night clubs'],
                  'golden house':['golden hous'],
                  'avalon':['avolon'],
                  'aylesbray lodge guest house':['aylesbray lodge'],
                  'leicester':['lecester'],
                  'asian oriental':['oriental'],
                  'allenbell':['allen bell'],
                  'bridge guest house':['bridge house'],

                  }
'''
fix_value_dict = {'guest house':['guesthouse'],
                  'cheap':['inexpensive','less expensive'],
                  'christ college':['christ s college','christ s'],
                  '02:45':['2:45'],
                  'cafe uno':['caffe uno'],
                  'el shaddia guesthouse':['el shaddai'],
                  'centre':['center'],
                  '08:00':['8:00'],
                  'peterborough':['peterbourgh'],
                  'the man on the moon':['man on the moon'],
                  'london liverpool street':['london liverpool'],
                  'swimming pool':['water slide'],
                  'multiple sports':['sports'],
                  'moderate':['reasonable','medium','reasonably'],
                  'kings college':['king s college'],
                  'saint johns college':['saint john s college'],
                  '04:15':['4:15'],
                  'kettles yard':['kettle s yard'],
                  'churchills college':['churchill college','churchhill college'],
                  '04:45':['4:45'],
                  '20:45':['20.45'],
                  'portugese':['portuguese'],
                  'a and b guest house':['a and b guesthouse','a and b quest house'],
                  'midsummer house restaurant':['midsummer house'],
                  'finches bed and breakfast':['finches bed & breakfast'],
                  'rosas bed and breakfast':['rosa s bed and breakfast'],
                  'pizza hut city centre':['pizza hut city','pizza hut'],
                  'emmanuel college':['emmauel college'],
                  'london kings cross':['london king s cross','kings cross station in london'],
                  'all saints church':['all saint s church'],
                  '16:45':['15:45'],
                  '08:45':['8:45'],
                  'pizza hut fenditton':['pizza hut fen ditton','pizza express fen ditton','fen ditton'],
                  '09:00':['9am','9 am'],
                  'north american':['american'],
                  'lovell lodge':['lovell lodege'],
                  'sheeps green and lammas land park fen causeway':['sheep s green and lammas land park fen causeway'],
                  'milton country park':['milton county park'],
                  '14:30':['2:30'],
                  'express by holiday inn cambridge':['express by holiday inn'],
                  '05:00':['5:00'],
                  'warkworth house':['warworth house'],
                  'acorn guest house':['acorn'],
                  '09:15':['9:15'],
                  'cambridge belfry':['caridge belfrey'],
                  'parkside pools':['parkside'],
                  'theatre':['theater'],
                  'dojo noodle bar':['dojo noddle bar'],
                  'holy trinity church':['holy trinity'],
                  '03:45':['3:45'],
                  'huntingdon marriott hotel':['huntington marriott hotel'],
                  '09:45':['9:45'],
                  'alexander bed and breakfast':['alexander bed and breafast'],
                  '07:30':['7:30'],
                  'king hedges learner pool':['kings hedges'],
                  '15:00':['1500'],
                  'hobsons house':['hobson s house'],
                  '01:30':['21:30'],
                  'norwich':['norwhich'],
                  '0':['zero'],
                  'australian':['australasian'],
                  'birmingham new street':['birmignham new street'],
                  'stazione restaurant and coffee bar':['stazione']
                  }
                  '''