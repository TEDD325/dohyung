/*******************************************************************************
 * Copyright 2016-2017 Dell Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
 * in compliance with the License. You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License
 * is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
 * or implied. See the License for the specific language governing permissions and limitations under
 * the License.
 *
 * @microservice: core-metadata
 * @author: Jim White, Dell
 * @version: 1.0.0
 *******************************************************************************/

package org.edgexfoundry.controller.integration;

import static org.edgexfoundry.test.data.ScheduleData.TEST_SCHEDULE_NAME;
import static org.edgexfoundry.test.data.ScheduleData.checkTestData;
import static org.edgexfoundry.test.data.ScheduleData.newTestInstance;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

import java.lang.reflect.Field;
import java.util.List;

import org.edgexfoundry.Application;
import org.edgexfoundry.controller.impl.ScheduleControllerImpl;
import org.edgexfoundry.dao.AddressableRepository;
import org.edgexfoundry.dao.ScheduleEventRepository;
import org.edgexfoundry.dao.ScheduleRepository;
import org.edgexfoundry.domain.meta.Addressable;
import org.edgexfoundry.domain.meta.Schedule;
import org.edgexfoundry.domain.meta.ScheduleEvent;
import org.edgexfoundry.exception.controller.DataValidationException;
import org.edgexfoundry.exception.controller.LimitExceededException;
import org.edgexfoundry.exception.controller.NotFoundException;
import org.edgexfoundry.exception.controller.ServiceException;
import org.edgexfoundry.test.category.RequiresMongoDB;
import org.edgexfoundry.test.category.RequiresSpring;
import org.edgexfoundry.test.category.RequiresWeb;
import org.edgexfoundry.test.data.AddressableData;
import org.edgexfoundry.test.data.ScheduleData;
import org.edgexfoundry.test.data.ScheduleEventData;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.experimental.categories.Category;
import org.junit.runner.RunWith;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.SpringApplicationConfiguration;
import org.springframework.test.context.junit4.SpringJUnit4ClassRunner;
import org.springframework.test.context.web.WebAppConfiguration;

@RunWith(SpringJUnit4ClassRunner.class)
@SpringApplicationConfiguration(classes = Application.class)
@WebAppConfiguration("src/test/resources")
@Category({RequiresMongoDB.class, RequiresSpring.class, RequiresWeb.class})
public class ScheduleControllerTest {

  private static final String LIMIT = "maxLimit";

  @Autowired
  private ScheduleRepository repos;

  @Autowired
  private ScheduleEventRepository scheduleEventRepos;

  @Autowired
  private AddressableRepository addrRepos;

  @Autowired
  private ScheduleControllerImpl controller;

  private String id;
  private String start;
  private String end;

  @Before
  public void setup() {
    Schedule schedule = ScheduleData.newTestInstance();
    start = schedule.getStart();
    end = schedule.getEnd();
    repos.save(schedule);
    id = schedule.getId();
  }

  @After
  public void cleanup() throws Exception {
    resetControllerMAXLIMIT();
    resetRepos();
    scheduleEventRepos.deleteAll();
    addrRepos.deleteAll();
    repos.deleteAll();
  }

  @Test
  public void testSchedule() {
    Schedule schedule = controller.schedule(id);
    checkTestData(schedule, id, start, end, ScheduleData.TEST_RUN_ONCE_FALSE);
  }

  @Test(expected = NotFoundException.class)
  public void testScheduleWithUnknownId() {
    controller.schedule("nosuchid");
  }

  @Test(expected = ServiceException.class)
  public void testScheduleException() throws Exception {
    unsetRepos();
    controller.schedule(id);
  }

  @Test
  public void testSchedules() {
    List<Schedule> schedules = controller.schedules();
    assertEquals("Find all not returning a list with one schedule", 1, schedules.size());
    checkTestData(schedules.get(0), id, start, end, ScheduleData.TEST_RUN_ONCE_FALSE);
  }

  @Test(expected = ServiceException.class)
  public void testSchedulesException() throws Exception {
    unsetRepos();
    controller.schedules();
  }

  @Test(expected = LimitExceededException.class)
  public void testSchedulesMaxLimitExceeded() throws Exception {
    unsetControllerMAXLIMIT();
    controller.schedules();
  }

  @Test
  public void testScheduleForName() {
    Schedule schedule = controller.scheduleForName(TEST_SCHEDULE_NAME);
    checkTestData(schedule, id, start, end, ScheduleData.TEST_RUN_ONCE_FALSE);
  }

  @Test(expected = NotFoundException.class)
  public void testScheduleForNameWithNoneMatching() {
    controller.scheduleForName("badname");
  }

  @Test(expected = ServiceException.class)
  public void testScheduleForNameException() throws Exception {
    unsetRepos();
    controller.scheduleForName(TEST_SCHEDULE_NAME);
  }

  @Test
  public void testAdd() {
    Schedule schedule = newTestInstance();
    schedule.setName("NewName");
    String newId = controller.add(schedule);
    assertNotNull("New device id is null", newId);
    assertNotNull("Modified date is null", schedule.getModified());
    assertNotNull("Create date is null", schedule.getCreated());
  }

  @Test(expected = DataValidationException.class)
  public void testAddWithBadCron() {
    Schedule schedule = newTestInstance();
    schedule.setName("NewName");
    schedule.setCron("badcron");
    controller.add(schedule);
  }

  @Test(expected = ServiceException.class)
  public void testAddNull() {
    controller.add(null);
  }

  @Test(expected = DataValidationException.class)
  public void testAddWithSameName() {
    Schedule schedule = repos.findOne(id);
    schedule.setId(null);
    controller.add(schedule);
  }

  @Test(expected = ServiceException.class)
  public void testAddException() throws Exception {
    unsetRepos();
    Schedule schedule = newTestInstance();
    schedule.setName("NewName");
    controller.add(schedule);
  }

  @Test
  public void testDelete() {
    assertTrue("Delete did not return correctly", controller.delete(id));
  }

  @Test(expected = NotFoundException.class)
  public void testDeleteWithNone() {
    controller.delete("badid");
  }

  @Test(expected = ServiceException.class)
  public void testDeleteException() throws Exception {
    unsetRepos();
    controller.delete(id);
  }

  @Test
  public void testDeleteByName() {
    assertTrue("Delete did not return correctly", controller.deleteByName(TEST_SCHEDULE_NAME));
  }

  @Test(expected = NotFoundException.class)
  public void testDeleteByNameWithNone() {
    controller.delete("badname");
  }

  @Test(expected = ServiceException.class)
  public void testDeleteByNameException() throws Exception {
    unsetRepos();
    controller.deleteByName(TEST_SCHEDULE_NAME);
  }

  @Test(expected = DataValidationException.class)
  public void testDeleteAssociatedToEvent() {
    Addressable addr = AddressableData.newTestInstance();
    addrRepos.save(addr);
    ScheduleEvent event = ScheduleEventData.newTestInstance();
    event.setAddressable(addr);
    scheduleEventRepos.save(event);
    controller.delete(id);
  }

  @Test
  public void testUpdate() {
    Schedule schedule = repos.findOne(id);
    schedule.setOrigin(1234);
    assertTrue("Update did not complete successfully", controller.update(schedule));
    Schedule schedule2 = repos.findOne(id);
    assertEquals("Update did not work correclty", 1234, schedule2.getOrigin());
    assertNotNull("Modified date is null", schedule2.getModified());
    assertNotNull("Create date is null", schedule2.getCreated());
    assertTrue("Modified date and create date should be different after update",
        schedule2.getModified() != schedule2.getCreated());
  }

  @Test(expected = DataValidationException.class)
  public void testUpdateWithBadCron() {
    Schedule schedule = repos.findOne(id);
    schedule.setCron("badcron");
    controller.update(schedule);
  }

  @Test
  public void testUpdateChangeNameWhileNotAssocToEvent() {
    Schedule schedule = repos.findOne(id);
    schedule.setName("newname");
    assertTrue("Update did not complete successfully", controller.update(schedule));
  }

  @Test(expected = DataValidationException.class)
  public void testUpdateChangeNameWhileAssocToEvent() {
    Addressable addr = AddressableData.newTestInstance();
    addrRepos.save(addr);
    ScheduleEvent event = ScheduleEventData.newTestInstance();
    event.setAddressable(addr);
    scheduleEventRepos.save(event);
    Schedule schedule = repos.findOne(id);
    schedule.setName("newname");
    controller.update(schedule);
  }

  @Test(expected = ServiceException.class)
  public void testUpdateException() throws Exception {
    unsetRepos();
    Schedule schedule = repos.findOne(id);
    schedule.setOrigin(1234);
    controller.update(schedule);
  }

  @Test(expected = NotFoundException.class)
  public void testUpdateWithNone() {
    Schedule schedule = repos.findOne(id);
    schedule.setId("badid");
    schedule.setName("badname");
    schedule.setOrigin(1234);
    controller.update(schedule);
  }

  private void unsetRepos() throws Exception {
    Class<?> controllerClass = controller.getClass();
    Field temp = controllerClass.getDeclaredField("repos");
    temp.setAccessible(true);
    temp.set(controller, null);
  }

  private void resetRepos() throws Exception {
    Class<?> controllerClass = controller.getClass();
    Field temp = controllerClass.getDeclaredField("repos");
    temp.setAccessible(true);
    temp.set(controller, repos);
  }

  private void unsetControllerMAXLIMIT() throws Exception {
    Class<?> controllerClass = controller.getClass();
    Field temp = controllerClass.getDeclaredField(LIMIT);
    temp.setAccessible(true);
    temp.set(controller, 0);
  }

  private void resetControllerMAXLIMIT() throws Exception {
    Class<?> controllerClass = controller.getClass();
    Field temp = controllerClass.getDeclaredField(LIMIT);
    temp.setAccessible(true);
    temp.set(controller, 1000);
  }
}
