/* Welcome to the SQL mini project. For this project, you will use
Springboard' online SQL platform, which you can log into through the
following link:

https://sql.springboard.com/
Username: student
Password: learn_sql@springboard

The data you need is in the "country_club" database. This database
contains 3 tables:
    i) the "Bookings" table,
    ii) the "Facilities" table, and
    iii) the "Members" table.

Note that, if you need to, you can also download these tables locally.

In the mini project, you'll be asked a series of questions. You can
solve them using the platform, but for the final deliverable,
paste the code for each solution into this script, and upload it
to your GitHub.

Before starting with the questions, feel free to take your time,
exploring the data, and getting acquainted with the 3 tables. */



/* Q1: Some of the facilities charge a fee to members, but some do not.
Please list the names of the facilities that do. */
SELECT name
FROM `Facilities` AS f
WHERE f.membercost > 0.0;


/* Q2: How many facilities do not charge a fee to members? */
SELECT COUNT( * )
FROM `Facilities` AS f
WHERE f.membercost = 0.0

/* Q3: How can you produce a list of facilities that charge a fee to members,
where the fee is less than 20% of the facility's monthly maintenance cost?
Return the facid, facility name, member cost, and monthly maintenance of the
facilities in question. */
SELECT f.facid, f.name, f.membercost, f.monthlymaintenance
FROM `Facilities` AS f
WHERE f.membercost > 0.0
AND f.membercost < 0.2 * f.monthlymaintenance
LIMIT 0 , 30

/* Q4: How can you retrieve the details of facilities with ID 1 and 5?
Write the query without using the OR operator. */
SELECT *
FROM `Facilities`
WHERE `facid`
IN ( 1, 5 ) 

/* Q5: How can you produce a list of facilities, with each labelled as
'cheap' or 'expensive', depending on if their monthly maintenance cost is
more than $100? Return the name and monthly maintenance of the facilities
in question. */
SELECT * ,
CASE
WHEN `monthlymaintenance` >100
THEN 'expensive'
ELSE 'cheap'
END AS `cheap_or_expensive`
FROM `Facilities`
LIMIT 0 , 30

/* Q6: You'd like to get the first and last name of the last member(s)
who signed up. Do not use the LIMIT clause for your solution. */

SELECT firstname, surname
FROM Members
WHERE joindate = (
SELECT MAX( joindate )
FROM Members ) 

/* Q7: How can you produce a list of all members who have used a tennis court?
Include in your output the name of the court, and the name of the member
formatted as a single column. Ensure no duplicate data, and order by
the member name. */
SELECT f.name AS court_name, CONCAT( m.firstname, ' ', m.surname ) AS member_name
FROM Bookings AS b
INNER JOIN Facilities AS f ON b.facid = f.facid
INNER JOIN Members AS m ON b.memid = m.memid
WHERE b.facid
IN ( 0, 1 ) 

/* Q8: How can you produce a list of bookings on the day of 2012-09-14 which
will cost the member (or guest) more than $30? Remember that guests have
different costs to members (the listed costs are per half-hour 'slot'), and
the guest user's ID is always 0. Include in your output the name of the
facility, the name of the member formatted as a single column, and the cost.
Order by descending cost, and do not use any subqueries. */
SELECT f.name, CONCAT( m.firstname, ' ', m.surname ) AS member_name,
CASE WHEN b.memid =0
THEN (
b.slots * f.guestcost
)
ELSE (
b.slots * f.membercost
)
END AS total_cost
FROM Bookings AS b
INNER JOIN Facilities AS f ON b.facid = f.facid
INNER JOIN Members AS m ON b.memid = m.memid
WHERE (

CASE WHEN b.memid =0
THEN (
b.slots * f.guestcost
)
ELSE (
b.slots * f.membercost
)
END
) >30 AND b.starttime LIKE '2012-09-14%'
ORDER BY total_cost DESC 

/* Q9: This time, produce the same result as in Q8, but using a subquery. */
SELECT CONCAT(m.firstname,' ',m.surname) AS member_name, sub.fname AS fname, sub.total_cost AS total_cost
FROM Members AS m
INNER JOIN (
    SELECT f.name as fname, CASE WHEN b.memid=0 THEN (b.slots*f.guestcost) ELSE (b.slots*f.membercost) END AS total_cost
    FROM Bookings as b
    INNER JOIN Facilities AS f ON b.facid= f.facid 
    WHERE (CASE WHEN b.memid =0
                THEN (
                        b.slots * f.guestcost
                        )
                ELSE (
                        b.slots * f.membercost
                        )
                END) >30 AND b.starttime LIKE '2012-09-14%'
) AS sub
ORDER BY total_cost DESC

/* Q10: Produce a list of facilities with a total revenue less than 1000.
The output of facility name and total revenue, sorted by revenue. Remember
that there's a different cost for guests and members! */
SELECT sub.facid, sub.name, sub.revenue
FROM (

SELECT f.facid AS facid, f.name AS name, SUM(
CASE WHEN b.memid =0
THEN b.slots * f.guestcost
ELSE b.slots * f.membercost
END ) AS revenue
FROM Facilities AS f
INNER JOIN Bookings AS b ON f.facid = b.facid
GROUP BY f.name
) AS sub
WHERE sub.revenue <1000