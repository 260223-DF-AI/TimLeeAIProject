-- More details in planning/about_database.md

DROP TABLE IF EXISTS drivers, images, http_requests, cv_results, llm CASCADE;
DROP TYPE IF EXISTS image_classes, prompt_types CASCADE;
CREATE TYPE image_classes AS ENUM('safe_driving', 'phone_usage', 'radio', 'drinking',
        'reaching', 'hair_makeup', 'turned_to_passenger');

CREATE TABLE drivers(
    driver_id VARCHAR(4) PRIMARY KEY,
    first_name VARCHAR(50),
    last_name VARCHAR(100),
    driver_age INTEGER
);

CREATE TABLE images(
    image_id SERIAL PRIMARY KEY,
    image_name VARCHAR(50) NOT NULL,
    image_class VARCHAR(2) NOT NULL,
    partition_loc VARCHAR(10) NOT NULL,
    driver_id VARCHAR(4) NOT NULL REFERENCES drivers(driver_id)
);


CREATE TABLE http_requests(
    http_id SERIAL PRIMARY KEY,
    time_received TIMESTAMP NOT NULL
);

CREATE TABLE cv_results(
    cv_id SERIAL PRIMARY KEY,
    http_id INTEGER NOT NULL REFERENCES http_requests(http_id),
    image_id INTEGER NOT NULL REFERENCES images(image_id),
    cv_result TEXT NOT NULL
);

CREATE TYPE prompt_types AS ENUM('image', 'summary', 'other');

CREATE TABLE llm(
    llm_id SERIAL PRIMARY KEY,
    http_id INTEGER NOT NULL REFERENCES http_requests(http_id),
    time_sent TIMESTAMP NOT NULL,
    time_received TIMESTAMP NOT NULL,
    prompt_type prompt_types NOT NULL,
    prompt_body TEXT NOT NULL,
    response_body TEXT NOT NULL
);
