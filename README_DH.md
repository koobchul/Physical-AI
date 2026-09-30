시점 문제 해결을 위한 체크리스트

1. 고개 돌리지 않고 몸만 평행하게 좌우이동 해보기
 시차가 실제처럼 느껴진다면 위치와 스케일 정상, 화각 문제
 시차가 없거나 약하면, 월드스케일 문제.

2. 늘어나는 현상이 종횡비 불일치에서 온다는 설
 python case_1_width_scale.py --width-scale 0.5
 이를 실행했을 때 세로로 길쭉한게 사라지고 정상 비율이 된다면 그것이 범인일 가능성이 높음.
 안 되면 반대 방향도 해보기
 python case_1_width_scale.py --width-scale 2.0
 2.0 looks wider
 
 20.0 looked even better


--------------------------------------------

변하는지 기록할 것

---------------------------------------------
 이어서 frustrum_width 조정
 python case_1_width_scale.py --fov-scale-x 0.5
 looks big and expanded(distance is distorted shorter)
 python case_1_width_scale.py --fov-scale-x 2.0
 (distance distorted longer)

 만약 위에선 변화했는데 아래 코드에선 변화가 없었다면 MuJoCo가 frustrum을 무시하고 종횡비로만 결정한다 - 화각을 뷰포트 크기로 제어하면 됨
 만약 아래에서만 변화한다면 값만 고치면 됨(쉬움)

 32 makes vertically longer

 ---------------------------------------------

 현실적인 조작을 위한 대안

 1. 양 팔 동시 조작 : 풋페달
 **시점 조정이 어려울 것으로 예상되므로, 시작 지점을 우리가 원하는 지점으로 설정할 필요가 있음.**
 
 풋페달로 실험할 제품 : Retro 108 Mechanical Keyboard - Dual Super Buttons(8Bitdo)
 왼쪽에 B, 오른쪽에 A버튼이 위치한 구조.
    B버튼은 그리퍼팔(왼팔)의 추종 ON/OFF 조작
    A버튼은 손팔(오른팔)의 추종 ON/OFF 조작
 
 양쪽 팔 모두 손 트래킹은 상시 활성화.
    왼손 엄지/검지로 그리퍼 조작 가능
    오른손 손 전체 트래킹으로 손 조작 가능.

   ----- 1번 플랜 시행 -----

   현재 CapsLock, ScrollLock으로 배정된 A,B 버튼의 배정을 풀어주기 위해
   xmodmap -e "keycode 66 = NoSymbol" -e "keycode 78 = NoSymbol"를 시행한 상태.
   만약 원래대로 되돌리고 싶다면
   setxkbmap -layout us(쓰고 있는 배열)을 시행하면 댐.

 2. 양 팔 동시 조작 : 음성조작
 **시점 조정이 어려울 것으로 예상되므로, 시작 지점을 우리가 원하는 지점으로 설정할 필요가 있음.**
 
 마이크를 이용한 ON/OFF : Faster-whisper등의 S2T 적용
    Left라고 말하면 왼팔 추종 ON/OFF 조작
    Right라고 말하면 오른팔 추종 ON/OFF 조작
    Left & Right라고 말하면 양팔 조작
 
 양쪽 팔 모두 손 트래킹은 상시 활성화.
    왼손 엄지/검지로 그리퍼 조작 가능
    오른손 손 전체 트래킹으로 손 조작 가능.

 3. 한 팔 조작 - 1번 후보
 **시점 조정이 어려울 것으로 예상되므로, 시작 지점을 우리가 원하는 지점으로 설정할 필요가 있음.**
 왼손 : 주먹으로 추종 ON/OFF
 오른손 : 손 모양 상시 트래킹, 손(손목) 위치에 따라 로봇 팔 조작

 4. 한 팔 조작 - 2번 후보
 **시점 조정이 어려울 것으로 예상되므로, 시작 지점을 우리가 원하는 지점으로 설정할 필요가 있음.**
 왼손 : 주먹으로 추종 ON/OFF, 손(손목) 위치에 따라 로봇 팔 조작
 오른손 : 손 모양 상시 트래킹. 팔의 움직임엔 영향을 주지 않음.