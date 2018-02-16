package main

import (
	//"bufio"
	"encoding/json"
	//"encoding/csv"
	//"errors"
	"fmt"
	"io/ioutil"
	"sort"
	//"io"
	//"log"
	//"math"
	"os"
	"path/filepath"
	//"golang.org/x/sync/syncmap"
	//"sort"
	"strconv"
	"strings"
	//"sync"
	"github.com/jinzhu/gorm"
	//"github.com/nlopes/slack"
)

func check(err error) {
	if err != nil {
		panic(err)
	}
}

/*
func splitByGender() {
	fmt.Println("starting to connect to DB...")
	db := ConnectToPG()
	fmt.Println("intermediate check")
	SetupDB(db)
	fmt.Println("connected")
	m2fFile, err := os.Create("maleToFemale" + ".txt")
	check(err)
	defer m2fFile.Close()
	m2mFile, err := os.Create("maleToMale" + ".txt")
	check(err)
	defer m2mFile.Close()
	var messages []Message
	db.Find(&messages)
	for _, message := range messages {
		index := strings.Index(message.Text, "<@U")
		if index != -1 &&
			!strings.Contains(message.Text, "uploaded a file") &&
			!strings.Contains(message.Text, "has left the channel") &&
			!strings.Contains(message.Text, "has joined the channel") {
			userId := message.Text[index+2 : index+11]
			var atUser User
			db.Where("slack_id = ?", userId).First(&atUser)
			var fromUser User
			db.Where("slack_id = ?", message.User).First(&fromUser)
			if atUser.Gender == true {
				m2fFile.WriteString(message.Text)
			} else if atUser.Gender == false && fromUser.Gender == false {
				m2mFile.WriteString(message.Text)
			} else {

			}
		}
	}
}*/

func main() {
	//latency()
	getAllMessages()
	removeUserTagsAndNewLines()
}

func removeUserTagsAndNewLines() {
	fileName := "spark.txt"
	dat, err := ioutil.ReadFile(fileName)
	check(err)
	allMsgs := string(dat)
	index := strings.Index(allMsgs, "<@U")
	for ; index != -1; index = strings.Index(allMsgs, "<@U") {
		fmt.Println(index)
		allMsgs = allMsgs[:index] + allMsgs[index+12:]
	}
	allMsgs = strings.Replace(allMsgs, "\n", " ", -1)
	fmt.Println(allMsgs)

	f, err := os.Create(fileName + "Clean" + ".txt")
	check(err)
	f.WriteString(allMsgs)
	f.Close()
}

func latency() {
	var latencyMeasures []float64
	fileObject, err := os.Open("py/direct_messages/neel.json")
	dat, err := ioutil.ReadAll(fileObject)
	check(err)
	messagesObject := Messages{}
	err = json.Unmarshal(dat, &messagesObject)
	messages := messagesObject.Messages
	check(err)
	prevUser := messages[0].User
	for i, message := range messages {
		currentUser := message.User
		if currentUser != prevUser {
			Request, err := strconv.ParseFloat(message.Ts, 64)
			check(err)
			Response, err := strconv.ParseFloat(messages[i-1].Ts, 64)
			check(err)
			latencyMeasure := Response - Request
			latencyMeasures = append(latencyMeasures, latencyMeasure)
			prevUser = currentUser
		}
	}
	sort.Float64s(latencyMeasures)
	medianIndex := len(latencyMeasures) / 2
	lowerQuartileIndex := len(latencyMeasures) / 4
	upperQuartileIndex := lowerQuartileIndex * 3
	fmt.Println("lower Q: " + strconv.FormatFloat(latencyMeasures[lowerQuartileIndex], 'f', -1, 64) +
		" median: " + strconv.FormatFloat(latencyMeasures[medianIndex], 'f', -1, 64) +
		" upperQuartile: " + strconv.FormatFloat(latencyMeasures[upperQuartileIndex], 'f', -1, 64))
}

func getAllMessages() {

	searchDir := "/Users/irfanfaizullabhoy/spark-data/"
	outputFile, err := os.Create("spark" + ".txt")
	defer outputFile.Close()
	check(err)
	fileList := []string{}
	var allMessages []Message

	err = filepath.Walk(searchDir, func(path string, f os.FileInfo, err error) error {
		fileList = append(fileList, path)
		return nil
	})
	check(err)
	for i, file := range fileList {
		if !(i == 0 || i == len(fileList)-1) {
			if strings.HasSuffix(file, ".json") && !strings.Contains(file, "integration_logs") {
				fmt.Println(file)
				fileObject, err := os.Open(file)
				dat, err := ioutil.ReadAll(fileObject)
				check(err)
				messages := []Message{}
				err = json.Unmarshal(dat, &messages)
				check(err)
				allMessages = append(allMessages, messages...)
				err = fileObject.Close()
				check(err)
			}
		}
	}
	for _, msg := range allMessages {
		outputFile.WriteString(msg.Text + " ")
	}
}

/*
func storeAllMessages(allMessages []Message) {
	fmt.Println("starting to connect to DB...")
	db := ConnectToPG()
	SetupDB(db)
	fmt.Println("intermediate check")
	SetupDB(db)
	fmt.Println("connected")

	userText := map[string]string{}
	for _, message := range allMessages {
		text := userText[message.User]
		userText[message.User] = text + " " + message.Text
	}

	fmt.Println(userText)

	for k, v := range userText {

		f, err := os.Create(k + ".txt")
		check(err)
		f.WriteString(v)
		f.Close()
	}
}

func createUsers() {
	fmt.Println("starting to connect to DB...")
	db := ConnectToPG()
	fmt.Println("intermediate check")
	SetupDB(db)
	fmt.Println("connected")
	api := slack.New("")
	users, err := api.GetUsers()
	check(err)
	for _, user := range users {
		fmt.Println(user.Name)
		db.Create(&User{SlackID: user.ID, Name: user.Name})
	}
}*/

type User struct {
	gorm.Model
	SlackID string `json:"slackid"`
	Name    string `json:"name"`
	Gender  bool   `json:"gender"`
	Status  int    `json:"int"`
}

type Messages struct {
	Messages []Message `json:"messages"`
}

type Message struct {
	MsgID     uint   `gorm:"primary_key"`
	Text      string `json:"text"`
	User      string `json:"user"`
	ChannelID string `json:"channel_id"`
	TeamID    string `json:"team_id"`
	Type      string `json:"type"`
	Ts        string `json:"ts"`
}
