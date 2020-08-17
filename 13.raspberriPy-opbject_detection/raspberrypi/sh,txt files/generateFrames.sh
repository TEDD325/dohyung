#!/bin/bash
######################################################################################
# mp4 비디오에서 이미지 추출하는 스크립트
# - 비디오 파일 하나 당 최대 999999 프레임까지만 추출 가능
# - 디렉토리 내에 들어 있는 모든 mp4 확장자 파일을 대상으로 함
# - 사용법: 
#   ./ 디렉토리 밑의 mp4 비디오에 대해
#   초당 5개의 이미지를 (즉, 매 200ms) 
#   720픽셀 사이즈로 변환하여 
#   <비디오파일명><일련번호>.<FORMAT> 파일로 추출하려면
#   아래 스크립트에서 변수값 변경
#
#  FPS="5"
#  DIRECTORY="./"
#  SCALE="scale="720:-1","
#  FORMAT=".png"
######################################################################################

FPS="5"
DIRECTORY="./"
SCALE="scale="720:-1","
#SCALE="scale="360:-1","
FORMAT=".png"


# 지정한 디렉토리 밑에 기존 이미지 파일들을 모두 지움
find "$DIRECTORY" -name "*.$FORMAT" | xargs rm -f

# mp4 파일 찾기
unset a i
while IFS= read -r -d '' file; do
  a[i++]="$file"
done < <(find "$DIRECTORY" -name '*.mp4' -type f -print0)

# 찾은 mp4 비디오파일들에 대해 ffmpeg 명령 수행
for n in "${a[@]}"
do
   :
   echo $n
   # 확장자 mp4 떼어내고
   FILEPREFIX=$(echo $n | sed 's/.mp4//g')
   # 각 jpg 파일에 이름 붙여 생성
   ffmpeg -i "$FILEPREFIX".mp4 -y -an -q 0 -vf "$SCALE"fps="$FPS" "$FILEPREFIX"%06d."$FORMAT"
done

# 기존 mp4 비디오 파일 지우려면 아래 주석 제거
find "$DIRECTORY" -name '*.mp4' | xargs rm -f
