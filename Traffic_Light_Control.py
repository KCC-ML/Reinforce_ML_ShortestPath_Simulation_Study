# .sumogcfg > .net.xml -> parsing -> modify 'duration' variable of traffic lights
# ML <------------------------------------ .sumocfg <------------------------------.
#            'duration' of vehicles                                                |
# ML ------------------------------------> Traffic_Light_Control.py -------> .net.xml
#        'duration' of traffic lights


# 학습한 신호등 duration 값 (ML 학습 모듈에서 받아오는 값)
new_durations = [20, 5, 20, 5]

# .net.xml 파싱 - 특정 id를 갖는 신호등의 duration 값을 사용자지정 phase 값으로 수정
from xml.etree.ElementTree import parse

tree = parse('C:/Program Files (x86)/Eclipse/Sumo/tools/map_created.net.xml')
root = tree.getroot()

traffic_light_logic_phases = root.findall(".//*[@id='gneJ1']/phase")
traffic_light_logic_phases[0].set('duration', str(new_durations[0]))
traffic_light_logic_phases[1].set('duration', str(new_durations[1]))
traffic_light_logic_phases[2].set('duration', str(new_durations[2]))
traffic_light_logic_phases[3].set('duration', str(new_durations[3]))

# 파일 출력을 sumo_home/tools로 하면 .sumocfg에 바로 적용 가능
tree.write('output.net.xml')